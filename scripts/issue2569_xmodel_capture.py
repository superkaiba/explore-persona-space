"""Issue #2569 leg 7 — paired same-text cross-model teacher-forced capture (plan v4 P-D).

Captures v_C (last prompt token) + v_A (answer-span mean) for the SAME 60,000 banked
conversations (20,000 holdout + 40,000 SAE-fit rows of the pinned #779 corpus,
split-sha-asserted through the reused #2476 staging) under BOTH models:

- Qwen/Qwen2.5-7B-Instruct  at layers 14/19/26 (28 layers, d=3584)
- meta-llama/Llama-3.1-8B-Instruct at layers 16/22/30 (32 layers, d=4096)

Ported pattern (``git show origin/issue-2378:scripts/issue2378_capture.py``, per
`.claude/rules/artifact-reuse.md` § Porting): the bf16-as-uint16 codec, the
offset-mapping span helpers (#2054 lineage), longest-first token-budget batch packing,
manual RIGHT padding + vectorized span-mask gathers, explicit ``.to(device)`` (never
``device_map="auto"``), introspection-guarded ``logits_to_keep=1``, and the chat-template
fingerprint assert. Storage is fp16 with a max-|x| < 65,504 assert per array, falling
back to the ported bit-exact bf16-as-uint16 codec on overflow (plan §4 leg 7 step 1).

Tokenization convention (matches the banked Qwen oracle, ``issue779_collect.py``):
``prompt_text = apply_chat_template(msgs, add_generation_prompt=True)`` and
``full_text = apply_chat_template(msgs + [assistant], add_generation_prompt=False)``;
the answer span is token positions ``[prompt_len, full_len)`` of the FULL render —
INCLUDING the trailing end-of-turn tokens — and v_C sits at ``prompt_len - 1``.
Both renders are tokenized with ``add_special_tokens=False`` (Qwen2.5's default adds
nothing — asserted equivalent at entry, oracle parity; Llama-3.1's default would
prepend a SECOND ``<|begin_of_text|>`` on top of the template's own). A per-row
prefix-seam assert (``full_ids[:prompt_len] == prompt_ids``) guards the BPE seam
(gotchas.md teacher-forced-capture rules); seam mismatches are DROPPED and counted,
never coerced.

hidden_states indexing convention (LOAD-BEARING — the #2569 P-D root cause). HF
returns ``len(hidden_states) == n_layers + 1``, where ``hidden_states[0]`` is the
EMBEDDING output and ``hidden_states[i]`` is the state AFTER i transformer blocks.
The banked #779 store numbers its layers 0-based BY BLOCK, so its layer L is the
state after block L — i.e. ``hidden_states[L + 1]``, NOT ``hidden_states[L]``. The
``--layers`` values here (14/19/26 Qwen, 16/22/30 Llama) are banked-store layer
numbers and are read with the +1 (``forward_batches``); ``_parse_layers`` bounds them
to ``[0, n_layers - 1]`` accordingly. The original implementation read
``hidden_states[layer]``, one block EARLIER than the banked data it must be
comparable with: the spot-gate measured rel_l2 0.4633 on v_C L14 against the banked
oracle (0.40 on v_A), and a layer-neighbour scan matched at offset +1 to
0.0122 / 0.0026 — bf16 noise. Any change to this convention must be re-derived
against the store, not inferred from the flag names.

Identity gates (plan §7 rows; BOTH are preconditions of the production pass):

- ``--phase identity-gate`` (B5, mandatory for Llama — the side with NO banked
  oracle): on 8 fixed rows spanning both corpora, recompute both summaries via a
  SECOND path that derives token boundaries from the tokenizer's offset mapping on
  the FULL concatenated sequence (independent of the capture path's
  prompt-render/full-render prefix logic) plus an unpadded batch-1 forward; assert
  identical token-boundary indices, identical shapes, and numerical agreement within
  fp16-scale tolerance. Any mismatch HALTS P-D (rc=4).
  Rationale: Qwen has the banked store as an external correctness oracle; Llama has
  none — a systematic Llama span/layer/boundary error degrades the v_C and v_A
  alignments TOGETHER, so the leg-7 correspondence test cannot discriminate it, and
  it would fire the registered leg-7 kill as a FALSE headline that is unrecoverable
  once the pod is terminated.
  B5 BLIND SPOT (disclosed per ``.claude/rules/smoke-blind-spots.md``): B5 does NOT
  certify the hidden_states LAYER CONVENTION documented above. Both of its arms call
  ``forward_batches``, so they share whatever convention that function implements —
  a layer-index error is invariant across the two arms and B5 PASSES on it BY
  CONSTRUCTION. Measured with the pre-fix off-by-one live: B5 on qwen passed at
  worst_rel_diff 0.0142 (48/48) while the spot-gate against the EXTERNAL banked
  oracle failed at 0.4633 on one of the same rows. B5 covers span / boundary /
  padding and batched-vs-unpadded numerics only. Consequence for the model with no
  oracle: a Llama-side layer-convention error is caught by NOTHING in this script —
  the qwen external-oracle spot-gate is the only layer-convention check, and it is
  trusted to transfer only because both models are read through the same
  ``forward_batches`` code path (the prior claim that B5 asserts layer indices was
  false: its assert compares shapes).
- ``--phase spot-gate`` (Qwen, G2-style, RETAINED): teacher-forced recompute vs the
  banked ``final_token_capture`` store on 8 rows.

Phases: ``select`` (row selection + corpus tags + text staging, stream-reduced with
per-file checkpoint/resume; the staging resume fingerprint pins SOURCE CONTENT —
blob-grain identity of the raw-completions + sampling-manifest files, staged at one
per-run-pinned repo revision — so a regenerated banked chunk at the same path
re-stages instead of resume-skipping over stale text;
select-stage-texts-resume-content-unpinned) → ``identity-gate``/``spot-gate``
(regime-keyed PASS resume-skip — the regime keys carry the selection TEXT-CONTENT
fingerprint, not just the ci identity, so regenerated prompts/responses re-run the
gate; fixed 8-row roster + full comparison cardinality asserted; every CONSUMER of
a gate PASS re-verifies the record's regime against its LIVE inputs before
honouring it — ``_require_gate`` + ``_GATE_BINDING_FIELDS``,
pd-gate-pass-not-bound-to-live-regime) → ``capture`` (chunked,
regime-keyed resume incl. the content fingerprint; the built-in pilot-gate report
is measured on FRESH forward rows ONLY — a resume that did no work can never
overwrite a recorded FAIL, and a zero-fresh-work smoke resume with NO record
fails loud — and the production-scale pass REQUIRES a smoke-scale pilot PASS
whose recorded ``capture_params`` EXACTLY match this run's execution shape,
plan §7) → ``finalize`` (per (model, summary, layer) ``{model}_{vc,va}_L{K}.pt``
bundles + realized-row asserts + gate/pilot preconditions bound to the chunk
store's OWN regime + HF upload) → ``sentinel`` (pod-side done JSON; REQUIRES both
models' finalize metas — the done-claim never precedes its outputs).

Observed banked schemas (probed 2026-08-25, keys only):
``raw_completions/shardSS_chunkCCCC.json`` = ``{"shard_index": int, "chunk": int,
"rows": [{"ci": int, "prompt": str, "response": str} x 500]}`` (1,936 files);
``sampling_manifest/part_NNNNN.jsonl`` rows = ``{"i": int, "corpus": str,
"stream_pos": int, "prompt": str}`` (87 parts + meta.json; ``i`` IS the conversation
index the raw-completions ``ci`` keys — verified by matching row 0) — NOTE the key is
``i``, not ``ci``. Scoped ``list_repo_tree`` entries for BOTH prefixes carry
``blob_id`` (git content address) + ``size`` (probed 2026-08-26 at revision
``e156135bc11c821f5ae76ee352ae715186b895c4``; e.g. ``shard00_chunk0000.json`` →
``blob_id 1d0a97845c7a3adc4db501e650e35929dad35b8d, size 1153322, lfs False``) —
the basis of the staging content fingerprint. Content hygiene: prompt/response
text is NEVER printed or logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + HF credentials BEFORE numpy/torch (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2569.xmodel")

TASK_ID = 2569
FP16_MAX = 65504.0
ROW_FLOOR_PRODUCTION = 50_000
SMOKE_ROWS_CEILING = 256  # runs at or under this row cap are pilot/smoke scale

N1M_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m"
RAW_COMPLETIONS_PREFIX = f"{N1M_PREFIX}/raw_completions"
SAMPLING_MANIFEST_PREFIX = f"{N1M_PREFIX}/sampling_manifest"
BANKED_CAPTURE_PREFIX = f"{N1M_PREFIX}/final_token_capture"
HF_XMODEL_PREFIX = "issue2569_theory/analysis_tensors/xmodel"

# Per-model constants (Llama facts measured live at plan time — plan §12 assumption 8 —
# and RE-ASSERTED against the loaded config at driver entry).
MODEL_SPECS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "n_layers": 28,
        "hidden": 3584,
        "default_layers": (14, 19, 26),
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "n_layers": 32,
        "hidden": 4096,
        "default_layers": (16, 22, 30),
    },
}
SLOTS = ("v_C", "v_A")
GATE_ROWS = 8  # fixed identity/spot gate row count (plan §7 rows)


# ---------------------------------------------------------------------------
# Storage codec (ported: issue2378_capture.py L99-120)
# ---------------------------------------------------------------------------


def encode_bf16_u16(t: torch.Tensor) -> np.ndarray:
    """bf16 tensor -> uint16 bit array (bit-exact; ported #2378 ``_encode_bf16``)."""
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous().view(torch.int16).cpu().numpy().view(np.uint16).copy()


def decode_bf16_u16(a: np.ndarray) -> torch.Tensor:
    """uint16 bit array -> torch bf16 tensor (bit-exact; ported #2378 ``decode_bf16``)."""
    assert a.dtype == np.uint16, a.dtype
    arr = np.ascontiguousarray(a).view(np.int16)
    return torch.from_numpy(arr.copy()).view(torch.bfloat16)


def encode_summary(x: np.ndarray) -> tuple[np.ndarray, str]:
    """fp32 array -> (stored array, codec): fp16 when max|x| < 65,504, else the
    ported bf16-as-uint16 codec (plan §4 leg 7 storage contract). Returns
    ``("fp16"|"bf16u16")``; asserts finiteness first (fail loud, never coerce)."""
    x32 = np.asarray(x, dtype=np.float32)
    assert np.isfinite(x32).all(), "non-finite activation summary (fail loud)"
    amax = float(np.abs(x32).max()) if x32.size else 0.0
    if amax < FP16_MAX:
        return x32.astype(np.float16), "fp16"
    return encode_bf16_u16(torch.from_numpy(x32).to(torch.bfloat16)), "bf16u16"


def decode_summary(a: np.ndarray, codec: str) -> np.ndarray:
    """Inverse of :func:`encode_summary` -> fp32 numpy."""
    if codec == "fp16":
        return np.asarray(a, dtype=np.float32)
    if codec == "bf16u16":
        return decode_bf16_u16(np.asarray(a)).to(torch.float32).numpy()
    raise ValueError(f"unknown codec {codec!r}")


# ---------------------------------------------------------------------------
# Span/position helpers (ported: issue2378_capture.py L128-159, #2054 lineage)
# ---------------------------------------------------------------------------


def _char_span_to_token_span(offsets, char_start: int, char_end: int) -> tuple[int, int]:
    """Token i in span iff [tok_lo, tok_hi) overlaps [char_start, char_end);
    zero-width offset rows skipped; (0, 0) = no overlap (ported verbatim)."""
    lo: int | None = None
    hi = 0
    for i, (tok_lo, tok_hi) in enumerate(offsets):
        if tok_hi <= tok_lo:  # zero-width rows (some specials)
            continue
        if tok_hi <= char_start:
            continue
        if tok_lo >= char_end:
            break
        if lo is None:
            lo = i
        hi = i + 1
    if lo is None:
        return (0, 0)
    return (lo, hi)


def _token_before_char(offsets, char_pos: int) -> int | None:
    """Index of the LAST token ending at or before ``char_pos``; None when no token
    ends before it (never coerce to 0 — ported verbatim)."""
    idx = -1
    for i, (tok_lo, tok_hi) in enumerate(offsets):
        if tok_hi <= tok_lo:
            continue
        if tok_hi <= char_pos:
            idx = i
    return None if idx < 0 else idx


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _texts_content_sha(texts: list[dict]) -> str:
    """Content fingerprint of the CONSUMED text rows — ci + corpus + prompt +
    response, in kept order (order is output-affecting: the gate roster and the
    capture slice are position-dependent). The members are FILE-READ strings
    (staged ``texts_kept.jsonl``), never recomputed floats, so the key is
    machine-stable (code-style float-last-bit rule). This is what pins resume
    keys to WHAT the rows contain, not just WHICH rows were selected
    (gpu-gate-resume-key-omits-text-content): changed prompts/responses at the
    same ci selection change the key. Text is hashed, never printed/logged
    (content hygiene)."""
    h = hashlib.sha256()
    for r in texts:
        h.update(
            json.dumps(
                {
                    "ci": int(r["ci"]),
                    "corpus": str(r["corpus"]),
                    "prompt": str(r["prompt"]),
                    "response": str(r["response"]),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        h.update(b"\n")
    return h.hexdigest()


def _sha_int64(a: np.ndarray) -> str:
    """sha256 of an int64 array's bytes (pinned integer dtype — machine-stable,
    unlike recomputed-float hashes; code-style float-last-bit rule)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()


def _atomic_json(path: Path, obj: dict) -> None:
    """JSON write through the shared process-unique atomic-replace primitive."""
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))


def _atomic_torch_save(obj: dict, path: Path) -> None:
    """torch.save through atomic_replace (#2336 process-unique tmp)."""
    with atomic_replace(path) as tmp:
        torch.save(obj, tmp)


def _meta(phase: str) -> dict:
    """Reproducibility metadata block (git commit + dirty flag + timestamps)."""
    prov = git_provenance()
    md = as_metadata_dict(prov, phase=phase)
    md["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return md


def split_target(rows: int) -> tuple[int, int]:
    """(n_holdout, n_sae_fit) for a total row target: 1:2 ratio (20,000/40,000 at
    the 60,000 production target; proportional at smoke scale, both nonzero)."""
    assert rows >= 3, f"row target {rows} too small to split 1:2"
    n_hold = rows // 3
    return n_hold, rows - n_hold


def _check_regime(dirpath: Path, regime: dict, wipe_globs: list[str], tag: str) -> None:
    """Regime-keyed resume guard: generating-parameters-only key (never recomputed
    float hashes). On mismatch, WIPE the stale outputs matching ``wipe_globs`` and
    write the new regime (loud); on match, resume is allowed."""
    dirpath.mkdir(parents=True, exist_ok=True)
    rpath = dirpath / "regime.json"
    if rpath.exists():
        prior = json.loads(rpath.read_text())
        if prior == regime:
            return
        stale = [p for g in wipe_globs for p in sorted(dirpath.glob(g))]
        logger.warning(
            "[%s] regime changed — wiping %d stale outputs under %s", tag, len(stale), dirpath
        )
        for p in stale:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    _atomic_json(rpath, regime)


# ---------------------------------------------------------------------------
# Phase: select — row selection + corpus tags + text staging (stream-reduced)
# ---------------------------------------------------------------------------


def _t24_namespace(args) -> argparse.Namespace:
    """Compose the reused #2476 namespace through ITS OWN argparse defaults (the
    reused-module Namespace contract, gotchas.md; mirrors issue2569_rowbattery's
    ``_t24_args``). Used ONLY for the sha-asserted split/scratch-meta staging."""
    import issue2476_turnavg_sae as T24

    ns = T24._parse_args(
        [
            "--phase",
            "assemble",
            "--out-root",
            str(Path(args.out_root) / "t24"),
            "--device",
            "cpu",
        ]
    )
    if ns.sae_dir is None:  # mirror T24.main's post-parse resolution
        ns.sae_dir = ns.out_root / "sae_cache"
    return ns


def select_rows(args) -> dict:
    """Select the capture rows (holdout + SAE-fit heads), sha-asserted via the
    reused #2476 staging; returns {row_index, ci, pools meta, drops}."""
    import issue2476_turnavg_sae as T24

    t24 = _t24_namespace(args)
    row_ci, _prov, pools = T24._load_scratch_meta(t24)  # sha-asserts sae_fit + holdout
    hold = np.sort(np.asarray(pools["holdout"], dtype=np.int64))
    sae = np.sort(np.asarray(pools["sae_fit"], dtype=np.int64))
    rows_target = int(args.rows) if int(args.rows) > 0 else 60_000
    n_hold, n_sae = split_target(rows_target)
    n_hold = min(n_hold, len(hold))
    n_sae = min(n_sae, len(sae))
    sel = np.concatenate([hold[:n_hold], sae[:n_sae]])
    assert len(np.unique(sel)) == len(sel), "holdout/sae_fit selection overlap (must be disjoint)"
    cis = np.asarray(row_ci, dtype=np.int64)[sel]
    drops = Counter()
    keep = cis >= 0
    drops["pass_b_no_text"] = int((~keep).sum())  # ci=-1 pass_b rows have no n1m text
    return {
        "row_index": sel[keep],
        "ci": cis[keep],
        "n_hold": int(n_hold),
        "n_sae": int(n_sae),
        "rows_target": rows_target,
        "drops": drops,
    }


def _entries_fingerprint(entries: list[tuple[str, str | None, int | None]]) -> str:
    """sha256 over sorted ``(path, blob_id, size)`` lines — the BLOB-grain content
    identity of the source file set. blob_id is git's content address, so the
    digest changes iff any listed file's CONTENT changes, and is STABLE under
    unrelated commits elsewhere in the repo."""
    h = hashlib.sha256()
    for path, blob_id, size in sorted(entries, key=lambda e: e[0]):
        h.update(f"{path}\t{blob_id}\t{size}\n".encode())
    return h.hexdigest()


def _source_snapshot(args) -> tuple[str, list[tuple[str, str | None, int | None]]]:
    """Resolve the source-data snapshot ONCE per select run: the data repo's
    current revision sha plus ONE scoped tree walk per source prefix at that
    revision, as ``(path, blob_id, size)`` entries covering BOTH the
    raw-completions chunks AND the sampling-manifest parts — both are content
    inputs of ``texts_kept.jsonl`` (prompts/responses from the former, corpus
    tags from the latter).

    Within-run consistency: every select-phase ``stage_hub_file`` call passes
    ``revision=`` this sha, so all staged files come from ONE snapshot (the
    #2061 revision=None paired-file snapshot-split trap). Cross-run resume
    keys on the BLOB-grain ``_entries_fingerprint``, NOT this repo revision:
    the shared data repo takes constant unrelated fleet commits, so a
    repo-revision pin in the resume fingerprint would restart the ~1,936-file
    stream on every unrelated upload (the too-strict/deadlock direction of
    select-stage-texts-resume-content-unpinned), while the blob fingerprint
    changes exactly when a source file's content is regenerated."""
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile

    api = HfApi()
    rev = str(api.repo_info(args.hf_data_repo, repo_type="dataset").sha)
    entries: list[tuple[str, str | None, int | None]] = []
    for prefix in (RAW_COMPLETIONS_PREFIX, SAMPLING_MANIFEST_PREFIX):

        def _walk(prefix=prefix):
            return [
                (e.path, getattr(e, "blob_id", None), getattr(e, "size", None))
                # HUB_VERIFY_RETRY_EXEMPT: raw list_repo_tree wrapped in hub._retry_upload below
                for e in api.list_repo_tree(
                    repo_id=args.hf_data_repo,
                    repo_type="dataset",
                    revision=rev,
                    recursive=True,
                    path_in_repo=prefix,
                )
                if isinstance(e, RepoFile)
            ]

        got = hub._retry_upload(_walk, what=f"list_repo_tree({args.hf_data_repo}/{prefix})")
        assert got, f"no files under {prefix} at revision {rev}"
        entries.extend(got)
    return rev, sorted(entries, key=lambda e: e[0])


def _stage_corpus_tags(args, selected_ci: np.ndarray, snapshot: tuple[str, list]) -> dict[int, str]:
    """Stream the sampling-manifest parts (one at a time, delete after read) and
    join corpus tags for the SELECTED conversation indices ONLY.

    Manifest rows carry ``i`` (NOT ``ci`` — observed schema, module docstring) +
    ``corpus``; text fields are never retained or logged. Files + revision come
    from the per-run ``_source_snapshot`` so the fingerprint and the staged
    bytes cannot diverge."""
    rev, entries = snapshot
    want = set(int(c) for c in selected_ci)
    tags: dict[int, str] = {}
    files = sorted(
        p
        for p, _blob, _size in entries
        if p.startswith(SAMPLING_MANIFEST_PREFIX + "/") and Path(p).name.startswith("part_")
    )
    assert files, f"no sampling-manifest parts under {SAMPLING_MANIFEST_PREFIX}"
    scratch = Path(args.out_root) / "stage_manifest"
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, repo_path in enumerate(files):
        dest = scratch / Path(repo_path).name
        hub.stage_hub_file(args.hf_data_repo, repo_path, dest, repo_type="dataset", revision=rev)
        with open(dest, encoding="utf-8") as fh:
            for line in fh:  # text-mode iteration, never .splitlines() (#825/#950)
                if not line.strip():
                    continue
                rec = json.loads(line)
                i = int(rec["i"])
                if i in want:
                    tags[i] = str(rec["corpus"]).lower()
        dest.unlink()  # stream-reduce: never hold the whole manifest locally
        print(
            f"[select] manifest part {k + 1}/{len(files)} tagged={len(tags)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
        if len(tags) == len(want):
            break
    return tags


def _stage_texts(
    args,
    selected_ci: np.ndarray,
    tags: dict[int, str],
    sel_meta: dict,
    snapshot: tuple[str, list],
) -> None:
    """Stream the raw-completions chunks (stage one file, keep selected rows,
    delete — stream-reduce) into ``texts_kept.jsonl`` with per-file checkpoint +
    fingerprint-gated resume (code-style external-stream presumption, #1092).
    The resume fingerprint pins SOURCE CONTENT via the snapshot's blob-grain
    ``_entries_fingerprint`` (select-stage-texts-resume-content-unpinned): a
    regenerated banked chunk or manifest part at the SAME path changes its
    blob_id, so the sidecar mismatches and the stream re-stages instead of
    resume-skipping over stale staged text — while unrelated repo commits
    leave the fingerprint untouched (resume stays satisfiable)."""
    rev, entries = snapshot
    out = Path(args.out_root)
    texts_path = out / "texts_kept.jsonl"
    sidecar = out / "texts_processed.json"
    fingerprint = {
        "prefix": RAW_COMPLETIONS_PREFIX,
        "selected_sha": _sha_int64(selected_ci),
        "n_selected": int(len(selected_ci)),
        "source_content_sha256": _entries_fingerprint(entries),
    }
    processed: dict = {"fingerprint": fingerprint, "files": [], "kept": 0, "drops": {}}
    if sidecar.exists():
        prior = json.loads(sidecar.read_text())
        if prior.get("fingerprint") == fingerprint and texts_path.exists():
            processed = prior
        else:
            logger.warning("[select] text-staging fingerprint changed — restarting stream")
            texts_path.unlink(missing_ok=True)
    done = set(processed["files"])
    want = set(int(c) for c in selected_ci)
    drops = Counter({k: int(v) for k, v in processed["drops"].items()})
    files = sorted(p for p, _blob, _size in entries if p.startswith(RAW_COMPLETIONS_PREFIX + "/"))
    assert files, f"no raw-completions chunks under {RAW_COMPLETIONS_PREFIX}"
    scratch = out / "stage_texts"
    scratch.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kept = int(processed["kept"])
    for k, repo_path in enumerate(files):
        name = Path(repo_path).name
        if name in done:
            continue
        dest = scratch / name
        hub.stage_hub_file(args.hf_data_repo, repo_path, dest, repo_type="dataset", revision=rev)
        obj = json.loads(dest.read_text())
        if "rows" not in obj and {"skipped", "n_skipped"} <= set(obj.keys()):
            # Shard-level SKIP manifests ride the same prefix as the rows chunks
            # (the real store carries shard{16..31}_skipped.json among the 1,920
            # shardNN_chunkNNNN.json files — measured 2026-08-25, final-round
            # smoke: the strict rows assert crashed at file 1021/1936). They
            # record generator-skipped rows (never captured, so never
            # selectable) and carry no text; count them LOUDLY and move on.
            drops["skip_manifest_files"] += 1
            dest.unlink()
            done.add(name)
            processed.update(files=sorted(done), kept=kept, drops=dict(drops))
            _atomic_json(sidecar, processed)
            print(f"[select] skip-manifest {name} ({k + 1}/{len(files)})", flush=True)
            continue
        assert set(obj.keys()) >= {"rows", "shard_index", "chunk"}, sorted(obj.keys())
        lines = []
        for r in obj["rows"]:
            ci = int(r["ci"])
            if ci not in want:
                continue
            if not str(r.get("response", "")).strip():
                drops["empty_response"] += 1
                continue
            if ci not in tags:
                drops["manifest_tag_missing"] += 1
                continue
            lines.append(
                json.dumps(
                    {
                        "ci": ci,
                        "corpus": tags[ci],
                        "prompt": r["prompt"],
                        "response": r["response"],
                    },
                    ensure_ascii=False,
                )
            )
        if lines:
            with open(texts_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        kept += len(lines)
        dest.unlink()
        done.add(name)
        processed.update(files=sorted(done), kept=kept, drops=dict(drops))
        _atomic_json(sidecar, processed)
        print(
            f"[select] texts file {k + 1}/{len(files)} kept={kept}/{len(want)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    n_missing = len(want) - kept - drops["empty_response"] - drops["manifest_tag_missing"]
    drops["text_missing"] = int(max(0, n_missing))
    sel_meta["drops"].update(drops)
    sel_meta["n_texts_kept"] = kept


def phase_select(args) -> None:
    """Row selection + corpus tags + stream-reduced text staging (checkpointed)."""
    print("[phase=select]", flush=True)
    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "selection_meta.json"
    sel = select_rows(args)
    sel_meta = {
        "rows_target": sel["rows_target"],
        "n_hold": sel["n_hold"],
        "n_sae": sel["n_sae"],
        "n_selected_with_text_ci": int(len(sel["ci"])),
        "selected_ci_sha256": _sha_int64(sel["ci"]),
        "drops": dict(sel["drops"]),
        "metadata": _meta("select"),
    }
    snapshot = _source_snapshot(args)  # ONE revision + blob listing for the whole select run
    tags = _stage_corpus_tags(args, sel["ci"], snapshot)
    _stage_texts(args, sel["ci"], tags, sel_meta, snapshot)
    # persist the selected (row_index, ci) join for downstream provenance
    with atomic_replace(out / "selection_rows.npz") as tmp, open(tmp, "wb") as fh:
        np.savez(fh, row_index=sel["row_index"], ci=sel["ci"])
    _atomic_json(meta_path, sel_meta)
    floor = ROW_FLOOR_PRODUCTION if sel["rows_target"] >= ROW_FLOOR_PRODUCTION else 1
    assert sel_meta["n_texts_kept"] >= floor, (
        f"kept texts {sel_meta['n_texts_kept']} < floor {floor} (plan §8) — refusing to capture"
    )
    print(f"[select] done kept={sel_meta['n_texts_kept']} drops={sel_meta['drops']}", flush=True)


def load_selection(args) -> list[dict]:
    """Load the staged texts (list of {ci, corpus, prompt, response} dicts)."""
    path = Path(args.out_root) / "texts_kept.jsonl"
    assert path.exists(), f"{path} missing — run --phase select first"
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:  # text-mode iteration (#825/#950 U+2028 rule)
            if line.strip():
                rows.append(json.loads(line))
    seen: set[int] = set()
    uniq = []
    for r in rows:
        if int(r["ci"]) not in seen:  # idempotent re-append guard
            seen.add(int(r["ci"]))
            uniq.append(r)
    return uniq


# ---------------------------------------------------------------------------
# Tokenization + template fingerprint
# ---------------------------------------------------------------------------


def _render(tok, prompt: str, response: str) -> tuple[str, str]:
    """(prompt_text, full_text) under the banked-oracle convention
    (issue779_collect.capture_context_vector / capture_answer_vector)."""
    msgs = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt_text, full_text


def template_probe(tok, model_key: str) -> dict:
    """Chat-template fingerprint + generation-suffix + specials probes (fail loud).

    - ``template_sha``: sha256 of the template source (the #2378 fingerprint idea).
    - ``gen_suffix``: derived content-independently as the byte suffix
      ``render(add_generation_prompt=True)`` appends over the ``False`` render —
      asserted per-row later (the oracle's GENERATION_SUFFIX assert, generalized).
    - Qwen: default tokenization must equal ``add_special_tokens=False`` (oracle
      parity — the banked capture tokenized with defaults).
    - Llama: the template's own leading BOS must appear EXACTLY once under
      ``add_special_tokens=False`` (the double-BOS hazard is the reason for False).
    """
    template_sha = hashlib.sha256((tok.chat_template or "").encode()).hexdigest()
    probe = "probe question for template fingerprint"
    msgs = [{"role": "user", "content": probe}]
    with_gen = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    without = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert with_gen.startswith(without), (
        "generation render is not prefix-stable over the no-generation render — "
        "conditional-block template (chat-template drift class, #2378 r13); "
        "this capture convention requires a prefix-stable template"
    )
    gen_suffix = with_gen[len(without) :]
    assert gen_suffix, "empty generation suffix (template probe failed)"
    ids_default = tok(with_gen)["input_ids"]
    ids_false = tok(with_gen, add_special_tokens=False)["input_ids"]
    if model_key == "qwen":
        assert ids_default == ids_false, (
            "Qwen tokenizer default specials != add_special_tokens=False — oracle "
            "parity broken (banked capture used defaults)"
        )
    else:
        bos = tok.bos_token_id
        assert bos is not None and ids_false[0] == bos and ids_false.count(bos) == 1, (
            "Llama render must carry EXACTLY one leading template BOS under "
            f"add_special_tokens=False (got first={ids_false[:1]}, count={ids_false.count(bos)})"
        )
    return {"template_sha": template_sha, "gen_suffix": gen_suffix}


def tokenize_rows(tok, rows: list[dict], gen_suffix: str, max_tokens: int):
    """ONE batched tokenization of prompt + full renders; per-row seam checks.

    Returns (kept records, drop Counter). Kept record fields: ci, corpus,
    input_ids (FULL render), n_tokens, prompt_len, v_C_pos, ans_lo, ans_hi.
    Drop reasons are named + counted, never coerced (#825 zero-width class)."""
    rendered = [_render_cached(tok, r["prompt"], r["response"]) for r in rows]
    enc_p = tok([p for p, _ in rendered], add_special_tokens=False)
    enc_f = tok([f for _, f in rendered], add_special_tokens=False)
    kept: list[dict] = []
    drops: Counter = Counter()
    for r, (prompt_text, _full_text), p_ids, f_ids in zip(
        rows, rendered, enc_p["input_ids"], enc_f["input_ids"]
    ):
        if not f_ids or not p_ids:
            drops["empty_tokenization"] += 1
            continue
        if len(f_ids) > max_tokens:
            drops["over_length"] += 1
            continue
        if not prompt_text.endswith(gen_suffix):
            drops["gen_suffix_mismatch"] += 1
            continue
        plen = len(p_ids)
        if len(f_ids) <= plen:
            drops["answer_span_empty"] += 1
            continue
        if f_ids[:plen] != p_ids:
            drops["prefix_seam_mismatch"] += 1  # BPE seam (gotchas.md) — drop, never coerce
            continue
        if plen < 2:
            drops["no_context_token"] += 1
            continue
        kept.append(
            {
                "ci": int(r["ci"]),
                "corpus": r["corpus"],
                "input_ids": list(f_ids),
                "n_tokens": len(f_ids),
                "prompt_len": plen,
                "v_C_pos": plen - 1,
                "ans_lo": plen,
                "ans_hi": len(f_ids),
            }
        )
    return kept, drops


def _render_cached(tok, prompt: str, response: str) -> tuple[str, str]:
    """Render helper (kept separate so the identity gate reuses the EXACT capture
    render; no caching state — the name records the shared-single-source intent)."""
    return _render(tok, prompt, response)


# ---------------------------------------------------------------------------
# Model load + batched forwards (ported: issue2378_capture.py L631-765)
# ---------------------------------------------------------------------------


def _load_model_ctx(args, spec: dict) -> dict:
    """bf16 model on an explicit device with fail-loud config + HBM asserts."""
    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable (fail loud)")
        free, total = torch.cuda.mem_get_info()
        need = int(args.min_free_hbm_gb * 2**30)
        if free < need:
            raise RuntimeError(
                f"HBM preflight failed: free={free / 2**30:.1f} GiB < required "
                f"{args.min_free_hbm_gb:.1f} GiB (total {total / 2**30:.1f} GiB)"
            )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=spec["revision"], dtype=torch.bfloat16
    )
    model.to(device)  # explicit placement — never device_map="auto" (#825)
    model.eval()
    tcfg = getattr(model.config, "text_config", model.config)
    if tcfg.num_hidden_layers != spec["n_layers"]:
        raise RuntimeError(
            f"num_hidden_layers={tcfg.num_hidden_layers}, expected {spec['n_layers']} "
            "(plan §12 assumption 8)"
        )
    if tcfg.hidden_size != spec["hidden"]:
        raise RuntimeError(f"hidden_size={tcfg.hidden_size}, expected {spec['hidden']}")
    import inspect

    params = inspect.signature(model.forward).parameters
    # logits are unread — skip full-vocab logits when the EXPLICIT param exists
    # (bare **kwargs does NOT count; gotchas.md logits_to_keep entry).
    lk = {"logits_to_keep": 1} if "logits_to_keep" in params else {}
    tok = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
    return {
        "model": model,
        "tok": tok,
        "device": device,
        "spec": spec,
        "logits_kwargs": lk,
        "pad_id": int(pad_id),
        "checked": False,
    }


def pack_batches(recs: list[dict], batch_tokens: int, max_batch_rows: int) -> list[list[int]]:
    """Longest-first length-bucket packing under a token budget (padding waste
    bounded; the longest bucket runs FIRST so OOM fails fast). ``--max-batch-rows``
    is a per-forward PACKING knob, NOT a total-row cap (plan §4 leg 7 step 1;
    ported #2378 ``_pack_batches`` with the row_id key replaced by ci)."""
    order = sorted(range(len(recs)), key=lambda i: (-recs[i]["n_tokens"], recs[i]["ci"]))
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for i in order:
        if not cur:
            cur = [i]
            cur_max = recs[i]["n_tokens"]
            continue
        if (len(cur) + 1) * cur_max > batch_tokens or len(cur) >= max_batch_rows:
            batches.append(cur)
            cur = [i]
            cur_max = recs[i]["n_tokens"]
        else:
            cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def forward_batches(args, mctx: dict, recs: list[dict], layers: list[int]) -> dict:
    """Teacher-forced forwards for one chunk; returns {layer: {slot: (n, d) fp32}}
    aligned with ``recs`` order. Manual RIGHT padding (positions index the unpadded
    prefix; causal mask + attention_mask keep pads inert). Ported #2378
    ``_forward_chunk`` (uint16-at-once storage replaced by fp32 accumulation; the
    fp16/codec decision is made once per finalized array)."""
    model = mctx["model"]
    dev = mctx["device"]
    spec = mctx["spec"]
    n = len(recs)
    hidden = spec["hidden"]
    out = {layer: {s: np.empty((n, hidden), dtype=np.float32) for s in SLOTS} for layer in layers}
    for batch in pack_batches(recs, args.batch_tokens, args.max_batch_rows):
        bsz = len(batch)
        t_max = max(recs[i]["n_tokens"] for i in batch)
        ids = torch.full((bsz, t_max), mctx["pad_id"], dtype=torch.long)
        mask = torch.zeros((bsz, t_max), dtype=torch.long)
        for j, ri in enumerate(batch):
            ln = recs[ri]["n_tokens"]
            ids[j, :ln] = torch.tensor(recs[ri]["input_ids"], dtype=torch.long)
            mask[j, :ln] = 1
        with torch.no_grad():
            res = model(
                input_ids=ids.to(dev),
                attention_mask=mask.to(dev),
                output_hidden_states=True,
                **mctx["logits_kwargs"],
            )
        hs = res.hidden_states
        if not mctx["checked"]:
            if len(hs) != spec["n_layers"] + 1:
                raise RuntimeError(
                    f"expected {spec['n_layers'] + 1} hidden states (embeddings + layers), "
                    f"got {len(hs)}"
                )
            assert hs[0].shape[-1] == hidden, tuple(hs[0].shape)
            mctx["checked"] = True
        hdev = hs[0].device
        rows_t = torch.arange(bsz, device=hdev)
        pos_c = torch.tensor([recs[ri]["v_C_pos"] for ri in batch], device=hdev)
        lo = torch.tensor([recs[ri]["ans_lo"] for ri in batch], device=hdev)
        hi = torch.tensor([recs[ri]["ans_hi"] for ri in batch], device=hdev)
        t_idx = torch.arange(t_max, device=hdev)
        span = (t_idx[None, :] >= lo[:, None]) & (t_idx[None, :] < hi[:, None])
        denom = span.sum(1).clamp_min(1).float()
        bidx = np.asarray(batch)
        for layer in layers:
            # hidden_states INDEXING CONVENTION (load-bearing; #2569 P-D root cause).
            # `hs` has n_layers+1 entries: hs[0] is the EMBEDDING output and hs[i] is
            # the state AFTER i transformer blocks. The banked #779 store numbers its
            # layers 0-based BY BLOCK, so its layer L is the state after block L, i.e.
            # hs[L + 1]. Reading hs[layer] here indexes one block EARLIER than the
            # banked data this capture must be comparable with: the #779 spot-gate then
            # measured rel_l2 ~0.46 on v_C / ~0.40 on v_A against the banked oracle,
            # while a layer-neighbour scan matched at offset +1 to ~0.012 / ~0.002
            # (bf16 noise). Do NOT drop the +1 without re-deriving against the store.
            h = hs[layer + 1]
            v_c = h[rows_t, pos_c].float()
            v_a = (h.float() * span[..., None]).sum(1) / denom[:, None]
            out[layer]["v_C"][bidx] = v_c.cpu().numpy()
            out[layer]["v_A"][bidx] = v_a.cpu().numpy()
        del res, hs
    return out


# ---------------------------------------------------------------------------
# Identity gates
# ---------------------------------------------------------------------------


def _gate_rows(
    texts: list[dict], tok, gen_suffix: str, max_tokens: int, n: int = GATE_ROWS
) -> tuple[list[dict], list[dict]]:
    """Fixed gate rows: the FIRST n/2 lmsys + n/2 wildchat rows (kept order,
    deterministic) that SURVIVE tokenization — a legitimately-dropped candidate
    (over-length etc.) is skipped, never a gate failure; falls back to any-corpus
    fill when a corpus is short. Returns (text rows, tokenized records), paired."""
    picked_texts: list[dict] = []
    picked_recs: list[dict] = []
    quota = {"lmsys": n - n // 2, "wildchat": n // 2}
    for pass_corpus in ("lmsys", "wildchat", None):
        for r in texts:
            if len(picked_texts) >= n:
                break
            if pass_corpus is not None and r["corpus"] != pass_corpus:
                continue
            if quota.get(r["corpus"], 0) <= 0 and pass_corpus is not None:
                continue
            if any(int(p["ci"]) == int(r["ci"]) for p in picked_texts):
                continue
            kept, _drops = tokenize_rows(tok, [r], gen_suffix, max_tokens)
            if not kept:
                continue  # legitimately dropped candidate — skip, never a gate failure
            picked_texts.append(r)
            picked_recs.append(kept[0])
            if pass_corpus is not None:
                quota[r["corpus"]] -= 1
    assert len(picked_texts) == n, (
        f"gate roster short: {len(picked_texts)}/{n} rows survive tokenization — "
        "plan §7 registers a fixed-cardinality gate row set; a partial roster must "
        "fail loud, never PASS over fewer cells (pd-gate-cardinality-unenforced)"
    )
    return picked_texts, picked_recs


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 difference ||a-b|| / max(||a||, eps)."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a64 - b64) / max(np.linalg.norm(a64), 1e-12))


GATE_HALT_RC = 4  # designed gate-refusal exit (artifact-routed halt, gotchas.md pilot-gate rule)


def _gate_halt(args, name: str, report: dict, err: BaseException) -> None:
    """Designed HALT on a gate mismatch: persist the FAIL record (measured values +
    the failing assert text) and exit the DISTINCT gate rc — never a bare rc=1 the
    dispatcher reads as an anonymous crash (gotchas.md pilot-gate routing rule)."""
    report["verdict"] = "FAIL"
    report["failure"] = str(err)
    report["metadata"] = _meta(name)
    gate_dir = Path(args.out_root) / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(gate_dir / f"{name}.json", report)
    print(f"[{name}] FAIL — HALT P-D: {err}", flush=True)
    sys.stdout.flush()
    raise SystemExit(GATE_HALT_RC)


def _require_layer_coverage(layers: list[int], banked_layers: list[int]) -> None:
    """Spot-gate admissibility: every gated layer must exist in the banked oracle —
    an empty intersection would yield a verdict over ZERO admissible comparisons,
    indistinguishable from a genuine PASS (pd-gate-cardinality-unenforced). Refuse
    loud BEFORE any verdict is computed."""
    assert set(layers) <= set(banked_layers), (
        f"--layers {layers} not covered by the banked oracle layers {banked_layers} — "
        "zero admissible comparisons would make the verdict vacuous (fail loud)"
    )


def _gate_regime(
    args, spec: dict, layers: list[int], texts: list[dict], extra: dict | None = None
) -> dict:
    """Generating-parameters-only regime key for the GPU gate phases' resume
    idempotency: covers every output-affecting input of the gate — model, layers,
    tolerance, tokenize/batch knobs, gate cardinality, the selection IDENTITY
    (ordered ci array sha) AND the selection text CONTENT
    (``texts_sha256`` over ci+corpus+prompt+response per row) — never a status
    string, never a recomputed-float hash (machine-stable, code-style
    float-last-bit rule). The ci sha alone pins only WHICH rows were selected;
    without the content sha, regenerated prompts/responses at the same ci
    selection would reuse a stale gate PASS
    (gpu-gate-resume-key-omits-text-content). ``device`` is a member because
    the gate measures THAT device's kernels — a cpu-measured PASS certifies
    nothing about the cuda path. This regime is ALSO the consumer-side binding
    reference: ``_require_gate`` compares the recorded regime against the live
    one on ``_GATE_BINDING_FIELDS`` (pd-gate-pass-not-bound-to-live-regime)."""
    cis = np.asarray([int(r["ci"]) for r in texts], dtype=np.int64)
    key = {
        "model_id": spec["model_id"],
        "layers": [int(x) for x in layers],
        "device": str(args.device),
        "gate_rel_tol": float(args.gate_rel_tol),
        "max_capture_tokens": int(args.max_capture_tokens),
        "batch_tokens": int(args.batch_tokens),
        "max_batch_rows": int(args.max_batch_rows),
        "gate_rows": GATE_ROWS,
        "selection_ci_sha256": _sha_int64(cis),
        "texts_sha256": _texts_content_sha(texts),
        "n_texts": int(len(cis)),
    }
    if extra:
        key.update(extra)
    return key


# The gate-regime subset a recorded gate PASS certifies for a CONSUMER
# (pd-gate-pass-not-bound-to-live-regime): compared by EXACT equality between
# the PASS record's ``regime`` and the consumer's LIVE regime. MUST match:
#   model_id            — the gate certifies THIS model's capture path;
#   layers              — the verdict is per (layer x slot) cell; a capture at
#                         ungated layers has zero certified cells there;
#   device              — the gate measured THIS device's kernels (a cpu PASS
#                         certifies nothing about the cuda path);
#   max_capture_tokens  — the certified token boundaries were computed under
#                         this truncation cap;
#   batch_tokens / max_batch_rows — the identity claim is batched == batch-1
#                         UNDER this packing geometry (padded-batch bf16
#                         numerics vary with shape);
#   gate_rows           — the registered fixed roster cardinality;
#   selection_ci_sha256 / n_texts — the gate rows were drawn deterministically
#                         from THIS selection identity;
#   texts_sha256        — the selection text CONTENT (the blocker: regenerated
#                         prompts/responses at the same cis mean the PASS was
#                         measured on OTHER text).
# ALLOWED to differ, with reasons:
#   gate_rel_tol — DIRECTIONAL, not exact: a PASS at the recorded tol certifies
#                  worst rel-diff <= that tol, so an equal-or-looser live bar is
#                  certified a fortiori; a TIGHTER live bar is refused (checked
#                  separately in _require_gate — exact equality here would
#                  needlessly refuse a deliberate tolerance loosening);
#   spot_chunk   — gate-internal oracle identity: WHICH banked chunk the spot
#                  gate compared against does not change what the PASS
#                  certifies for the consumer; it stays in the record for the
#                  gate phase's OWN resume idempotency + audit (binding it
#                  would refuse a capture/finalize run whose --spot-chunk argv
#                  it never consumes — the too-strict/deadlock direction).
_GATE_BINDING_FIELDS = (
    "model_id",
    "layers",
    "device",
    "max_capture_tokens",
    "batch_tokens",
    "max_batch_rows",
    "gate_rows",
    "selection_ci_sha256",
    "texts_sha256",
    "n_texts",
)


def _gate_resume_skip(args, name: str, regime: dict) -> bool:
    """GPU gate phases are resume-idempotent (gpu-gate-phases-not-idempotent): a
    re-entry with a recorded PASS under an IDENTICAL regime is a no-op — no model
    load, no forwards, no record mutation. A FAIL record, a regime drift, or a
    regime-less legacy record re-runs the REAL gate (a fresh measurement of real
    forwards may then legitimately overwrite the record)."""
    path = Path(args.out_root) / "gates" / f"{name}.json"
    if not path.exists():
        return False
    rec = json.loads(path.read_text())
    if rec.get("verdict") == "PASS" and rec.get("regime") == regime:
        print(f"[{name}] resume-skip: PASS record present under an identical regime", flush=True)
        return True
    return False


def phase_identity_gate(args) -> None:
    """B5 identity gate (plan §7 row): independent-boundary + batch-1 recompute vs
    the capture path on 8 fixed rows; ANY mismatch HALTS P-D (rc=4).

    Independence: the second path derives token boundaries from the tokenizer's
    OFFSET MAPPING on the FULL concatenated sequence (``_char_span_to_token_span``
    + ``_token_before_char`` at the prompt/full char boundary) — it never uses the
    capture path's prompt-render prefix logic — and recomputes both summaries with
    an UNPADDED batch-1 forward."""
    spec = MODEL_SPECS[args.model]
    layers = _parse_layers(args, spec)
    print(f"[phase=identity-gate] model={args.model} layers={layers}", flush=True)
    texts = load_selection(args)
    regime = _gate_regime(args, spec, layers, texts)
    if _gate_resume_skip(args, f"identity_gate_{args.model}", regime):
        return
    mctx = _load_model_ctx(args, spec)
    tok = mctx["tok"]
    probe = template_probe(tok, args.model)
    rows, kept = _gate_rows(texts, tok, probe["gen_suffix"], args.max_capture_tokens)
    captured = forward_batches(args, mctx, kept, layers)

    report = {
        "model": args.model,
        "layers": layers,
        "rows": [],
        "rel_tol": args.gate_rel_tol,
        "regime": regime,
    }
    worst = 0.0
    n_cmp = 0
    try:
        for j, (r, rec) in enumerate(zip(rows, kept)):
            prompt_text, full_text = _render(tok, r["prompt"], r["response"])
            enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            assert ids == rec["input_ids"], "full-render ids differ across tokenization calls"
            lo, hi = _char_span_to_token_span(offsets, len(prompt_text), len(full_text))
            v_c_ind = _token_before_char(offsets, len(prompt_text))
            # identical token-boundary indices (B5 assert 1)
            assert lo == rec["ans_lo"] and hi == rec["ans_hi"], (
                f"row ci={rec['ci']}: offset-derived span ({lo},{hi}) != capture-path "
                f"({rec['ans_lo']},{rec['ans_hi']}) — HALT P-D (B5)"
            )
            assert v_c_ind is not None and v_c_ind == rec["v_C_pos"], (
                f"row ci={rec['ci']}: offset-derived v_C pos {v_c_ind} != capture-path "
                f"{rec['v_C_pos']} — HALT P-D (B5)"
            )
            # unpadded batch-1 forward (independent numeric path)
            one = dict(rec)
            vals = forward_batches(args, mctx, [one], layers)
            row_rep = {"ci": rec["ci"], "corpus": rec["corpus"], "diffs": {}}
            for layer in layers:
                for slot in SLOTS:
                    a = captured[layer][slot][j]
                    b = vals[layer][slot][0]
                    # identical layer indices + shapes (B5 asserts 2+3)
                    assert a.shape == b.shape == (spec["hidden"],), (a.shape, b.shape)
                    d = _rel_l2(a, b)
                    n_cmp += 1
                    worst = max(worst, d)
                    row_rep["diffs"][f"{slot}_l{layer}"] = d
                    assert d <= args.gate_rel_tol, (
                        f"row ci={rec['ci']} {slot} L{layer}: rel diff {d:.4g} > tol "
                        f"{args.gate_rel_tol} — HALT P-D (B5 numeric agreement)"
                    )
            report["rows"].append(row_rep)
    except AssertionError as err:
        report["worst_rel_diff"] = worst
        report["n_comparisons"] = n_cmp
        _gate_halt(args, f"identity_gate_{args.model}", report, err)
    expected = GATE_ROWS * len(layers) * len(SLOTS)
    assert n_cmp == expected > 0, (
        f"identity-gate comparisons {n_cmp} != expected {expected} — a verdict over a "
        "partial/vacuous cell set is inadmissible (pd-gate-cardinality-unenforced)"
    )
    report["n_comparisons"] = n_cmp
    report["expected_comparisons"] = expected
    report["worst_rel_diff"] = worst
    report["verdict"] = "PASS"
    report["template_sha"] = probe["template_sha"]
    report["metadata"] = _meta("identity-gate")
    gate_dir = Path(args.out_root) / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(gate_dir / f"identity_gate_{args.model}.json", report)
    print(f"[identity-gate] PASS model={args.model} worst_rel_diff={worst:.4g}", flush=True)


def phase_spot_gate(args) -> None:
    """Qwen G2-style spot gate (RETAINED, plan §7): teacher-forced recompute vs the
    banked ``final_token_capture`` store on 8 rows; mismatch HALTS (rc=4).

    The banked bundle (issue779_ffc_n1m_generate_capture ``_stack_chunk``) carries
    ``cx_last``/``v_x`` of shape (n, 3, H) at layers (14, 19, 26) + a ``ci`` list;
    gate rows are drawn FROM the staged bundle (intersected with the selected text
    rows) so no chunk search is needed."""
    assert args.model == "qwen", "spot-gate compares against the banked QWEN store"
    spec = MODEL_SPECS["qwen"]
    layers = _parse_layers(args, spec)
    print(f"[phase=spot-gate] layers={layers}", flush=True)
    texts_list = load_selection(args)
    regime = _gate_regime(
        args, spec, layers, texts_list, extra={"spot_chunk": str(args.spot_chunk)}
    )
    if _gate_resume_skip(args, "spot_gate_qwen", regime):
        return
    texts = {int(r["ci"]): r for r in texts_list}
    scratch = Path(args.out_root) / "stage_spot"
    scratch.mkdir(parents=True, exist_ok=True)
    dest = scratch / Path(args.spot_chunk).name
    hub.stage_hub_file(
        args.hf_data_repo, f"{BANKED_CAPTURE_PREFIX}/{args.spot_chunk}", dest, repo_type="dataset"
    )
    bundle = torch.load(dest, map_location="cpu", weights_only=False)
    banked_layers = [int(x) for x in bundle["layers"]]
    assert banked_layers == list(spec["default_layers"]), banked_layers
    _require_layer_coverage(layers, banked_layers)
    banked_ci = [int(c) for c in bundle["ci"]]
    overlap = [(pos, ci) for pos, ci in enumerate(banked_ci) if ci in texts]
    assert overlap, "no banked-chunk rows overlap the selection — pass a different --spot-chunk"
    mctx = _load_model_ctx(args, spec)
    probe = template_probe(mctx["tok"], "qwen")
    picks: list[tuple[int, int]] = []
    kept: list[dict] = []
    for pos, ci in overlap:  # keep only rows that survive tokenization (deterministic)
        if len(picks) >= GATE_ROWS:
            break
        k1, _d = tokenize_rows(
            mctx["tok"], [texts[ci]], probe["gen_suffix"], args.max_capture_tokens
        )
        if k1:
            picks.append((pos, ci))
            kept.append(k1[0])
    assert len(picks) == GATE_ROWS, (
        f"spot-gate roster short: {len(picks)}/{GATE_ROWS} banked-overlap rows survive "
        f"tokenization (overlap={len(overlap)}) — pass a different --spot-chunk or widen "
        "the selection; a partial roster must fail loud (pd-gate-cardinality-unenforced)"
    )
    captured = forward_batches(args, mctx, kept, layers)
    report = {
        "rows": [],
        "rel_tol": args.gate_rel_tol,
        "chunk": args.spot_chunk,
        "regime": regime,
    }
    worst = 0.0
    n_cmp = 0
    try:
        for j, (pos, ci) in enumerate(picks):
            row_rep = {"ci": ci, "diffs": {}}
            for li, layer in enumerate(banked_layers):
                if layer not in layers:
                    continue
                banked = {
                    "v_C": bundle["cx_last"][pos, li].to(torch.float32).numpy(),
                    "v_A": bundle["v_x"][pos, li].to(torch.float32).numpy(),
                }
                for slot in SLOTS:
                    d = _rel_l2(captured[layer][slot][j], banked[slot])
                    n_cmp += 1
                    worst = max(worst, d)
                    row_rep["diffs"][f"{slot}_l{layer}"] = d
                    assert d <= args.gate_rel_tol, (
                        f"spot-gate ci={ci} {slot} L{layer}: rel diff {d:.4g} > tol "
                        f"{args.gate_rel_tol} vs the banked oracle — HALT P-D"
                    )
            report["rows"].append(row_rep)
    except AssertionError as err:
        report["worst_rel_diff"] = worst
        report["n_comparisons"] = n_cmp
        _gate_halt(args, "spot_gate_qwen", report, err)
    expected = len(picks) * len(layers) * len(SLOTS)
    assert n_cmp == expected > 0, (
        f"spot-gate comparisons {n_cmp} != expected {expected} — a verdict over a "
        "partial/vacuous cell set is inadmissible (pd-gate-cardinality-unenforced)"
    )
    report["n_comparisons"] = n_cmp
    report["expected_comparisons"] = expected
    report["worst_rel_diff"] = worst
    report["verdict"] = "PASS"
    report["metadata"] = _meta("spot-gate")
    gate_dir = Path(args.out_root) / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(gate_dir / "spot_gate_qwen.json", report)
    dest.unlink()
    print(f"[spot-gate] PASS worst_rel_diff={worst:.4g}", flush=True)


# ---------------------------------------------------------------------------
# Phase: capture (chunked, regime-keyed resume, pilot-gate report)
# ---------------------------------------------------------------------------


def _parse_layers(args, spec: dict) -> list[int]:
    """Layer list from --layers csv (default: the model's registered triple)."""
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layers = list(spec["default_layers"])
    assert layers, (
        "--layers resolved to an EMPTY layer list — every gate/capture over it "
        "would be vacuous (pd-gate-cardinality-unenforced; fail loud)"
    )
    for layer in layers:
        # Layers are 0-based BLOCK indices matching the banked #779 store: layer L is
        # read as hidden_states[L + 1] (see forward_batches). The top nominal layer is
        # therefore n_layers - 1, whose read hs[n_layers] is the final block's output.
        assert 0 <= layer <= spec["n_layers"] - 1, (layer, spec["n_layers"])
    return layers


# The capture-parameter subset a pilot PASS certifies (compared by EXACT
# equality between the pilot record and the run about to proceed). These are
# the fields that define the MEASURED per-row execution shape — the plan §7
# P-D pilot row's own contract ("production entrypoint, production shape —
# same --batch-tokens 8192 --max-batch-rows 64 packing, same capture path")
# and the gotchas.md pilot-timing rule (measure at the sweep's execution
# shape). EXEMPT, with reasons: rows_cap + kept_ci_sha256 + texts_sha256 (the
# pilot is DELIBERATELY the smoke-scale row slice, so the row count and the
# slice-identity/content fingerprints differ by design); template_sha
# (throughput-inert — correctness drift is owned by the chunk-store regime
# wipe + the identity/spot gates); chunk_rows (checkpoint granularity: it
# never changes the forward shape, and at pilot scale — rows <=
# SMOKE_ROWS_CEILING < the chunk_rows default — it is inert by construction,
# so comparing it would pin a knob the pilot cannot exercise).
_PILOT_BINDING_FIELDS = (
    "model_id",
    "layers",
    "max_capture_tokens",
    "batch_tokens",
    "max_batch_rows",
    "device",
)


def _pilot_params(args, spec: dict, layers: list[int]) -> dict:
    """The execution-shape parameters a pilot measurement certifies (the
    ``_PILOT_BINDING_FIELDS`` values for THIS run's args). Computable from
    args+spec alone so preconditions can refuse BEFORE any model load."""
    return {
        "model_id": spec["model_id"],
        "layers": [int(x) for x in layers],
        "max_capture_tokens": int(args.max_capture_tokens),
        "batch_tokens": int(args.batch_tokens),
        "max_batch_rows": int(args.max_batch_rows),
        "device": str(args.device),
    }


def _pilot_binding_from_regime(regime: dict) -> dict:
    """Extract the pilot-binding subset from a chunk-store regime (what the
    capture ACTUALLY ran with — finalize binds to that, not to its own argv).
    A regime lacking the fields predates the binding contract: fail loud."""
    missing = [k for k in _PILOT_BINDING_FIELDS if k not in regime]
    assert not missing, (
        f"chunk-store regime lacks pilot-binding fields {missing} — a pre-binding "
        "chunk store cannot be verified against the pilot record "
        "(pd-pilot-pass-not-bound-to-production-regime); re-run --phase capture"
    )
    return {k: regime[k] for k in _PILOT_BINDING_FIELDS}


def _capture_regime(
    args, spec: dict, layers: list[int], kept_ci: np.ndarray, tsha: str, texts_sha: str
) -> dict:
    """Generating-parameters-only regime key for the chunk store (float-hash-free).
    Carries the selection IDENTITY (``kept_ci_sha256``) AND the consumed text
    CONTENT (``texts_sha256`` — file-read strings, so changed prompts/responses
    at the same ci slice wipe the stale chunks instead of resume-skipping over
    them; resume-keys-omit-content-and-required-outputs), plus ``device``
    (numerics differ across devices, so cpu- and cuda-computed chunks must never
    silently mix in one store). Embeds ``_pilot_params`` so the pilot-binding
    subset is readable back from ``regime.json`` at finalize."""
    return {
        **_pilot_params(args, spec, layers),
        "rows_cap": int(args.rows),
        "kept_ci_sha256": _sha_int64(kept_ci),
        "texts_sha256": texts_sha,
        "template_sha": tsha,
        "chunk_rows": int(args.chunk_rows),
    }


def _require_gate(args, name: str, live_regime: dict) -> None:
    """Assert a PASS gate record exists for this out-root AND that the PASS was
    measured under THIS run's live regime (a PRECONDITION of the production
    pass — plan §7; smoke-scale runs enforce it too, same path). Binding =
    exact equality on ``_GATE_BINDING_FIELDS`` (see the constant's enumeration
    of must-match vs allowed-to-differ members) plus the DIRECTIONAL tolerance
    rule: recorded gate_rel_tol <= live gate_rel_tol — a PASS at a looser bar
    does not certify a tighter live one. The tolerance rule fires only when
    the live regime CARRIES a tolerance (capture: argv-derived; finalize:
    deliberately absent — ``_gate_binding_from_store``). An unbound
    (regime-less) PASS is refused loud: a historical PASS whose inputs are
    unknown certifies nothing (pd-gate-pass-not-bound-to-live-regime)."""
    path = Path(args.out_root) / "gates" / f"{name}.json"
    assert path.exists(), f"gate record {path} missing — run the gate phase first (plan §7)"
    rec = json.loads(path.read_text())
    assert rec.get("verdict") == "PASS", f"{path}: verdict {rec.get('verdict')!r} != PASS"
    recorded = rec.get("regime")
    assert recorded is not None, (
        f"{path}: PASS record carries no regime — an UNBOUND gate PASS cannot certify the "
        "live inputs (pd-gate-pass-not-bound-to-live-regime); re-run the gate phase"
    )
    diffs = {
        k: {"recorded": recorded.get(k), "live": live_regime.get(k)}
        for k in _GATE_BINDING_FIELDS
        if recorded.get(k) != live_regime.get(k)
    }
    if diffs:
        raise AssertionError(
            f"{path}: gate PASS was measured under a DIFFERENT regime than this run's live "
            f"inputs — mismatched fields: {diffs} — re-run the gate phase "
            "(pd-gate-pass-not-bound-to-live-regime)"
        )
    live_tol = live_regime.get("gate_rel_tol")
    if live_tol is not None:
        rec_tol = recorded.get("gate_rel_tol")
        assert rec_tol is not None, (
            f"{path}: gate regime carries no gate_rel_tol — not a producer-written regime; "
            "re-run the gate phase (pd-gate-pass-not-bound-to-live-regime)"
        )
        assert float(rec_tol) <= float(live_tol), (
            f"{path}: gate PASS was measured at tolerance {rec_tol} LOOSER than this run's "
            f"{live_tol} — it does not certify the tighter live bar; re-run the gate phase "
            "(pd-gate-pass-not-bound-to-live-regime)"
        )


# Chunk-store regime fields that double as the finalize-side gate-binding knobs
# (all written by _capture_regime via _pilot_params, plus layers).
_GATE_STORE_KNOB_FIELDS = (
    "model_id",
    "layers",
    "device",
    "max_capture_tokens",
    "batch_tokens",
    "max_batch_rows",
)


def _gate_binding_from_store(args, chunk_regime: dict) -> dict:
    """Live gate-binding reference at FINALIZE
    (pd-gate-pass-not-bound-to-live-regime): execution-shape knobs come from
    the chunk store's OWN regime — what the capture ACTUALLY ran with, never
    finalize's argv (the pilot-binding principle,
    pd-pilot-pass-not-bound-to-production-regime) — while selection identity +
    content are RECOMPUTED from the staged texts NOW on disk (the FULL
    selection, mirroring the gate phases' own regime computation, so a
    texts_kept.jsonl regenerated after the gate ran refuses here), and the
    roster cardinality comes from the code constant. ``gate_rel_tol`` is
    deliberately ABSENT: finalize has no non-argv live tolerance, and the
    directional tolerance rule already bound at capture time where the argv IS
    the live regime. A chunk regime lacking the knob fields predates the
    binding contract: fail loud."""
    missing = [k for k in _GATE_STORE_KNOB_FIELDS if k not in chunk_regime]
    assert not missing, (
        f"chunk-store regime lacks gate-binding fields {missing} — a pre-binding chunk "
        "store cannot be verified against the gate record "
        "(pd-gate-pass-not-bound-to-live-regime); re-run --phase capture"
    )
    texts = load_selection(args)
    cis = np.asarray([int(r["ci"]) for r in texts], dtype=np.int64)
    return {
        **{k: chunk_regime[k] for k in _GATE_STORE_KNOB_FIELDS},
        "gate_rows": GATE_ROWS,
        "selection_ci_sha256": _sha_int64(cis),
        "texts_sha256": _texts_content_sha(texts),
        "n_texts": int(len(cis)),
    }


def _require_pilot_pass(args, pilot_params: dict) -> None:
    """Pilot-gate precondition (plan §7 P-D pilot row; pd-gate-precondition-bypass):
    the production-scale pass and finalize refuse to proceed until a smoke-scale
    pilot record with verdict PASS exists for this model AND that PASS was
    measured under THIS run's execution shape — exact equality on the
    ``_PILOT_BINDING_FIELDS`` (pd-pilot-pass-not-bound-to-production-regime; an
    unrelated historical PASS certifies nothing about the production regime).
    The smoke-scale capture itself is EXEMPT — it IS the pilot
    (gate-calibration parity, the #1345 class)."""
    path = Path(args.out_root) / "gates" / f"pilot_gate_{args.model}.json"
    assert path.exists(), (
        f"pilot gate record {path} missing — run the smoke-scale capture pilot "
        f"(0 < --rows <= {SMOKE_ROWS_CEILING}) first (plan §7)"
    )
    rec = json.loads(path.read_text())
    assert rec.get("verdict") == "PASS", (
        f"{path}: verdict {rec.get('verdict')!r} != PASS — the pilot gate blocks this phase"
    )
    recorded = rec.get("capture_params")
    assert recorded is not None, (
        f"{path}: PASS record carries no capture_params — an UNBOUND pilot PASS cannot "
        "certify the production regime (pd-pilot-pass-not-bound-to-production-regime); "
        "re-run the smoke-scale pilot at the production shape"
    )
    if recorded != pilot_params:
        diffs = {
            k: {"pilot": recorded.get(k), "production": pilot_params.get(k)}
            for k in sorted(set(recorded) | set(pilot_params))
            if recorded.get(k) != pilot_params.get(k)
        }
        raise AssertionError(
            f"{path}: pilot PASS was measured under a DIFFERENT capture regime than this "
            f"run — mismatched fields: {diffs} — re-run the smoke-scale pilot at the "
            "production shape (pd-pilot-pass-not-bound-to-production-regime)"
        )


def _capture_gate_name(args) -> str:
    """Gate consumed by capture/finalize for the selected model/text regime.

    The original shared-Qwen-text arm keeps the banked Qwen spot oracle.  A
    different answer-writer arm cannot compare its v_A to that oracle, so it
    must use the independent-boundary identity gate instead.  The default is
    unchanged for every historical invocation.
    """
    if args.model == "llama":
        return "identity_gate_llama"
    return "identity_gate_qwen" if args.qwen_gate == "identity" else "spot_gate_qwen"


def phase_capture(args) -> None:
    """Chunked teacher-forced capture for one model (regime-keyed resume; per-chunk
    atomic .pt checkpoints; per-unit progress lines; built-in pilot-gate report)."""
    spec = MODEL_SPECS[args.model]
    layers = _parse_layers(args, spec)
    print(f"[phase=capture] model={args.model} layers={layers} rows={args.rows}", flush=True)
    smoke_scale = 0 < int(args.rows) <= SMOKE_ROWS_CEILING
    pilot_params = _pilot_params(args, spec, layers)
    if not smoke_scale and not args.skip_gate_check:
        # plan §7 P-D pilot row: the production pass proceeds ONLY on a pilot PASS
        # measured under THIS run's execution shape (pd-gate-precondition-bypass +
        # pd-pilot-pass-not-bound-to-production-regime). The smoke-scale run IS
        # the pilot — exempt (a self-precondition would make it unrunnable).
        _require_pilot_pass(args, pilot_params)
    texts = load_selection(args)
    if not args.skip_gate_check:
        # B5/G2 PRECONDITION of the pass, BOUND to the live regime: the recorded
        # PASS must have been measured over THIS out-root's staged selection
        # (content included) at THIS run's execution shape. The live regime is
        # computed over the FULL selection BEFORE the --rows truncation,
        # mirroring the gate phases' own computation
        # (pd-gate-pass-not-bound-to-live-regime). Runs before any model load.
        gate_name = _capture_gate_name(args)
        _require_gate(args, gate_name, _gate_regime(args, spec, layers, texts))
    if int(args.rows) > 0:
        texts = texts[: int(args.rows)]
    assert texts, (
        "selection produced ZERO rows — refusing a vacuous capture "
        "(empty-selection fail-loud rule, gotchas.md)"
    )
    mctx = _load_model_ctx(args, spec)
    tok = mctx["tok"]
    probe = template_probe(tok, args.model)
    kept_ci = np.asarray([int(r["ci"]) for r in texts], dtype=np.int64)
    chunk_dir = Path(args.out_root) / "chunks" / args.model
    regime = _capture_regime(
        args, spec, layers, kept_ci, probe["template_sha"], _texts_content_sha(texts)
    )
    _check_regime(chunk_dir, regime, ["chunk*.pt"], f"capture/{args.model}")

    n_chunks = (len(texts) + args.chunk_rows - 1) // args.chunk_rows
    total_drops: Counter = Counter()
    t0 = time.time()
    rows_done = 0
    fresh_rows = 0  # rows computed by REAL forwards THIS process (pilot basis)
    fresh_wall_s = 0.0
    resumed_chunks = 0
    for k in range(n_chunks):
        cpath = chunk_dir / f"chunk{k:05d}.pt"
        chunk_texts = texts[k * args.chunk_rows : (k + 1) * args.chunk_rows]
        if cpath.exists():
            prior = torch.load(cpath, map_location="cpu", weights_only=False)
            total_drops.update(prior["drops"])
            rows_done += len(prior["ci"])
            resumed_chunks += 1
            print(f"[capture] chunk {k + 1}/{n_chunks} resume-skip", flush=True)
            continue
        c0 = time.time()
        kept, drops = tokenize_rows(tok, chunk_texts, probe["gen_suffix"], args.max_capture_tokens)
        total_drops.update(drops)
        arrays: dict[str, np.ndarray] = {}
        codecs: dict[str, str] = {}
        if kept:
            captured = forward_batches(args, mctx, kept, layers)
            for layer in layers:
                for slot in SLOTS:
                    key = f"{slot.lower().replace('_', '')}_l{layer}"  # vc_l14 / va_l14
                    arr, codec = encode_summary(captured[layer][slot])
                    arrays[key] = arr
                    codecs[key] = codec
        _atomic_torch_save(
            {
                "ci": np.asarray([r["ci"] for r in kept], dtype=np.int64),
                "corpus": [r["corpus"] for r in kept],
                "n_tokens": np.asarray([r["n_tokens"] for r in kept], dtype=np.int64),
                "prompt_len": np.asarray([r["prompt_len"] for r in kept], dtype=np.int64),
                "layers": layers,
                "arrays": arrays,
                "codecs": codecs,
                "drops": dict(drops),
                "regime": regime,
            },
            cpath,
        )
        rows_done += len(kept)
        fresh_rows += len(kept)
        fresh_wall_s += time.time() - c0
        print(
            f"[capture] chunk {k + 1}/{n_chunks} model={args.model} kept={len(kept)} "
            f"drops={dict(drops)} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    # Required-output presence check (resume-keys-omit-content-and-required-outputs):
    # the capture phase's declared outputs are EVERY chunk file — a resume-skip is
    # honoured per chunk only via cpath.exists() above, and the phase may not
    # report done with any chunk absent.
    missing_chunks = [k for k in range(n_chunks) if not (chunk_dir / f"chunk{k:05d}.pt").is_file()]
    assert not missing_chunks, (
        f"capture required outputs missing under {chunk_dir}: chunks {missing_chunks} "
        f"of {n_chunks} — refusing to report the phase done"
    )
    _pilot_gate_report(args, pilot_params, fresh_rows, fresh_wall_s, resumed_chunks)
    print(
        f"[capture] done model={args.model} realized={rows_done} drops={dict(total_drops)}",
        flush=True,
    )


def _pilot_gate_report(
    args, pilot_params: dict, fresh_rows: int, fresh_wall_s: float, resumed_chunks: int
) -> None:
    """Built-in pilot-gate report (plan §7 P-D pilot row): the 32-row smoke IS the
    measured sizing pilot. The measurement basis is FRESH forward-pass rows ONLY —
    resume-skipped chunks contribute NOTHING to per_row_s, so a relaunch that did
    no work can neither write nor overwrite the gate record: a recorded FAIL
    stands byte-identical and RE-HALTS (rc=3) at smoke scale
    (pilot-gate-halt-erased-by-resume), and a smoke-scale zero-fresh-work resume
    with NO record at all FAILS LOUD — the pilot verdict is the smoke capture's
    REQUIRED OUTPUT, and a resume-skip may not stand in for the measurement
    (resume-keys-omit-content-and-required-outputs). A fresh smoke-scale
    measurement writes the verdict WITH the ``capture_params`` it was measured
    under (``_pilot_params`` — what ``_require_pilot_pass`` later binds against;
    pd-pilot-pass-not-bound-to-production-regime); extrapolation > 2x the booked
    wall EXITS rc=3 (halt-and-report). Production-scale runs (rows == 0 or >
    SMOKE_ROWS_CEILING) write an INFORMATIONAL ``capture_wall_{model}.json``
    instead and NEVER touch the pilot gate record (the smoke verdict is the
    durable gate; sunk production compute is not re-gated on its own wall)."""
    smoke_scale = 0 < int(args.rows) <= SMOKE_ROWS_CEILING
    gate_dir = Path(args.out_root) / "gates"
    gate_path = gate_dir / f"pilot_gate_{args.model}.json"
    if fresh_rows <= 0 or fresh_wall_s <= 0:
        if gate_path.exists():
            prior = json.loads(gate_path.read_text())
            print(
                f"[pilot-gate] no fresh rows this run (resumed_chunks={resumed_chunks}) — "
                f"record unchanged (verdict={prior.get('verdict')})",
                flush=True,
            )
            if smoke_scale and prior.get("verdict") == "FAIL":
                print(
                    "[pilot-gate] recorded FAIL stands — the resume did no fresh work; "
                    "halt (rc=3, plan §7)",
                    flush=True,
                )
                sys.stdout.flush()
                raise SystemExit(3)
        elif smoke_scale:
            raise RuntimeError(
                f"[pilot-gate] required output {gate_path} is ABSENT after a "
                f"zero-fresh-work smoke resume (resumed_chunks={resumed_chunks}) — the "
                "resume-skip cannot stand in for the pilot measurement; wipe the chunk "
                f"store ({Path(args.out_root) / 'chunks' / args.model}) and re-run the "
                "smoke capture (resume-keys-omit-content-and-required-outputs)"
            )
        else:
            print(
                f"[pilot-gate] no fresh rows this run (resumed_chunks={resumed_chunks}) "
                "and no prior record — nothing to measure (production scale; the wall "
                "report is informational-only)",
                flush=True,
            )
        return
    per_row_s = fresh_wall_s / fresh_rows
    extrapolated_h = per_row_s * 60_000 * 2 / 3600.0
    booked_h = float(args.pilot_booked_wall_h)
    rec = {
        "per_row_s": per_row_s,
        "rows_measured": int(fresh_rows),
        "fresh_wall_s": float(fresh_wall_s),
        "resumed_chunks": int(resumed_chunks),
        "extrapolated_wall_h": extrapolated_h,
        "booked_wall_h": booked_h,
        "capture_params": pilot_params,
        "metadata": _meta("pilot-gate"),
    }
    if not smoke_scale:
        rec["informational"] = True
        _atomic_json(gate_dir / f"capture_wall_{args.model}.json", rec)
        print(
            f"[capture-wall] per_row_s={per_row_s:.3f} extrapolated_wall_h(60k x 2 models)="
            f"{extrapolated_h:.2f} booked_h={booked_h:.1f} — informational (production scale)",
            flush=True,
        )
        return
    verdict = "PASS" if extrapolated_h <= 2.0 * booked_h else "FAIL"
    rec["verdict"] = verdict
    print(
        f"[pilot-gate] per_row_s={per_row_s:.3f} extrapolated_wall_h(60k x 2 models)="
        f"{extrapolated_h:.2f} booked_h={booked_h:.1f} verdict={verdict}",
        flush=True,
    )
    _atomic_json(gate_path, rec)
    if verdict == "FAIL":
        print("[pilot-gate] halt-and-report: extrapolation > 2x booked wall (plan §7)", flush=True)
        sys.stdout.flush()
        raise SystemExit(3)


# ---------------------------------------------------------------------------
# Phase: finalize — per (model, summary, layer) bundles + realized-row asserts
# ---------------------------------------------------------------------------


def phase_finalize(args) -> None:
    """Assemble per (summary, layer) ``{model}_{vc,va}_L{K}.pt`` bundles from the
    chunk store; assert the realized-row bookkeeping identity + the production
    floor (plan §4 leg 7 step 1); upload the bundle set to the HF data repo."""
    spec = MODEL_SPECS[args.model]
    print(f"[phase=finalize] model={args.model}", flush=True)
    chunk_dir = Path(args.out_root) / "chunks" / args.model
    chunks = sorted(chunk_dir.glob("chunk*.pt"))
    assert chunks, f"no chunks under {chunk_dir} — run --phase capture first"
    regime = json.loads((chunk_dir / "regime.json").read_text())
    layers = [int(x) for x in regime["layers"]]
    if not args.skip_gate_check:
        # finalize is DOWNSTREAM of the gates: chunks exist even from a halted
        # pilot run (the chunk lands before the gate fires), so a blind finalize
        # must refuse rather than bundle+upload past a FAIL/missing verdict
        # (pd-gate-precondition-bypass). The pilot binding is checked against the
        # chunk store's OWN regime — what the capture ACTUALLY ran with — never
        # finalize's argv (pd-pilot-pass-not-bound-to-production-regime); the
        # gate binding composes the same store knobs with the selection content
        # recomputed from the staged texts NOW on disk
        # (pd-gate-pass-not-bound-to-live-regime).
        _require_gate(
            args,
            _capture_gate_name(args),
            _gate_binding_from_store(args, regime),
        )
        _require_pilot_pass(args, _pilot_binding_from_regime(regime))

    ci_parts, corpus_parts, drops = [], [], Counter()
    per_key: dict[str, list[np.ndarray]] = {}
    for cpath in chunks:
        c = torch.load(cpath, map_location="cpu", weights_only=False)
        assert c["regime"] == regime, f"{cpath}: chunk regime drift"
        drops.update(c["drops"])
        if len(c["ci"]) == 0:
            continue
        ci_parts.append(np.asarray(c["ci"], dtype=np.int64))
        corpus_parts.extend(c["corpus"])
        for key, arr in c["arrays"].items():
            per_key.setdefault(key, []).append(decode_summary(arr, c["codecs"][key]))
    ci = np.concatenate(ci_parts) if ci_parts else np.zeros(0, dtype=np.int64)
    realized = int(len(ci))
    n_selected = _n_selected(args)
    n_drops = int(sum(drops.values()))
    # bookkeeping identity: realized == selected-with-text minus counted capture drops
    assert realized == n_selected - n_drops, (realized, n_selected, dict(drops))
    floor = (
        ROW_FLOOR_PRODUCTION
        if int(args.rows) in (0,) or int(args.rows) >= ROW_FLOOR_PRODUCTION
        else GATE_ROWS
    )
    assert realized >= floor, (
        f"realized rows {realized} < floor {floor} (plan §8) — refusing to finalize"
    )
    assert len(np.unique(ci)) == realized, "duplicate ci in finalized store"

    final_dir = Path(args.out_root) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for layer in layers:
        for slot, tag in (("v_C", "vc"), ("v_A", "va")):
            key = f"{tag}_l{layer}"
            x32 = np.concatenate(per_key[key], axis=0)
            assert x32.shape == (realized, spec["hidden"]), x32.shape
            arr, codec = encode_summary(x32)
            name = f"{args.model}_{tag}_L{layer}.pt"
            _atomic_torch_save(
                {
                    "x": arr,
                    "codec": codec,
                    "ci": ci,
                    "corpus": list(corpus_parts),
                    "layer": int(layer),
                    "slot": slot,
                    "model_id": spec["model_id"],
                    "template_sha": regime["template_sha"],
                    "drops": dict(drops),
                    "n_selected_texts": n_selected,
                    "metadata": _meta("finalize"),
                },
                final_dir / name,
            )
            names.append(name)
    _atomic_json(
        final_dir / f"{args.model}_finalize_meta.json",
        {
            "model": args.model,
            "realized": realized,
            "n_selected_texts": n_selected,
            "drops": dict(drops),
            "layers": layers,
            "files": names,
            "ci_sha256": _sha_int64(ci),
            "metadata": _meta("finalize"),
        },
    )
    # Required-output presence check: every declared finalize output must be a
    # FILE on disk before the phase reports done / uploads
    # (resume-keys-omit-content-and-required-outputs; is_file per the
    # FILE-vs-DIR kind rule, gotchas.md).
    required = [*names, f"{args.model}_finalize_meta.json"]
    missing_out = [n for n in required if not (final_dir / n).is_file()]
    assert not missing_out, (
        f"finalize required outputs missing under {final_dir}: {missing_out} — "
        "refusing to report the phase done"
    )
    print(f"[finalize] realized={realized} files={len(names)}", flush=True)
    if not args.skip_upload:
        _upload_final(args, final_dir, required)


def _n_selected(args) -> int:
    """Selected-with-text row count this capture consumed (rows cap applied)."""
    texts = load_selection(args)
    return len(texts[: int(args.rows)]) if int(args.rows) > 0 else len(texts)


def _upload_final(args, final_dir: Path, names: list[str]) -> None:
    """ONE bulk upload_folder commit + exact-set presence verify (fail loud)."""
    url = hub._upload_folder_filtered(
        final_dir,
        repo_id=args.hf_data_repo,
        repo_type="dataset",
        path_in_repo=args.hf_prefix,
        allow_patterns=list(names),
        expected_repo_paths=[f"{args.hf_prefix}/{n}" for n in names],
    )
    if not url:  # fail-soft "" -> fail loud here (n1m convention)
        raise RuntimeError(
            f"bulk upload of {len(names)} files to {args.hf_prefix} returned no URL"
        )
    print(f"[finalize] uploaded {len(names)} files -> {args.hf_prefix}", flush=True)


# ---------------------------------------------------------------------------
# Phase: sentinel (pod-side done JSON; the poller drains it into markers)
# ---------------------------------------------------------------------------


def phase_sentinel(args) -> None:
    """Write the pod-side done sentinel (pod-side-reporting contract). The
    dispatcher composes WHEN this runs; this phase only writes the JSON +
    emits the terminal ``[phase=done]`` line."""
    assert args.sentinel_path, "--sentinel-path required for --phase sentinel"
    payload = {
        # poll_pipeline._SENTINEL_REQUIRED_KEYS envelope (schema pinned to
        # poll_pipeline.SENTINEL_SCHEMA_VERSION_SUPPORTED = 1): without it the
        # VM drain warn-skips the file every tick and never renames it to
        # .processed (verified by a _parse_sentinel dry-run, final-round smoke
        # 2026-08-25). Mirrors issue779_common.write_sentinel's envelope; the
        # explicit path (the plan-declared /workspace/logs/issue-2569-pd-done
        # .json) is kept instead of the helper's derived filename.
        "sentinel_schema_version": 1,
        "kind": "phase-pd-done",
        "version": 1,
        "blocks_pipeline": False,
        "note": "P-D cross-model capture done",
        "issue": TASK_ID,
        "phase": "done",
        "status": "ok",
        "rc": 0,
        "out_root": str(args.out_root),
        "metadata": _meta("sentinel"),
    }
    for model in MODEL_SPECS:
        # The done sentinel is the P-D lane's COMPLETION claim (plan §4: the
        # gates are preconditions of the production sentinel) — both models'
        # finalize outputs are REQUIRED before it may be written; a silently
        # optional read here would signal done with a model's finalize absent
        # (resume-keys-omit-content-and-required-outputs).
        meta = Path(args.out_root) / "final" / f"{model}_finalize_meta.json"
        assert meta.is_file(), (
            f"required output {meta} missing — the done sentinel may not be written "
            f"before --phase finalize completed for {model} (plan §4: P-D captures "
            "BOTH models; a done-claim with a finalize output absent is a false done)"
        )
        m = json.loads(meta.read_text())
        payload[f"{model}_realized"] = m["realized"]
    path = Path(args.sentinel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)
    print("[phase=done]", flush=True)  # workflow-lint: phase-done-reserved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PHASES = {
    "select": phase_select,
    "identity-gate": phase_identity_gate,
    "spot-gate": phase_spot_gate,
    "capture": phase_capture,
    "finalize": phase_finalize,
    "sentinel": phase_sentinel,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI (argparse — per-issue phase-dispatch driver convention, code-style.md)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=sorted(PHASES), default=None)
    ap.add_argument("--import-check", action="store_true", help="argcheck + exit 0")
    ap.add_argument("--model", choices=sorted(MODEL_SPECS), default="qwen")
    ap.add_argument(
        "--rows",
        type=int,
        default=0,
        help="TOTAL row cap (0 = all/60,000 target; smoke: 32). NOT a packing knob.",
    )
    ap.add_argument(
        "--batch-tokens", type=int, default=8192, help="per-forward token budget (packing)"
    )
    ap.add_argument(
        "--max-batch-rows", type=int, default=64, help="per-forward row cap (packing, NOT total)"
    )
    ap.add_argument("--chunk-rows", type=int, default=2000)
    ap.add_argument("--max-capture-tokens", type=int, default=8192)
    ap.add_argument("--layers", default="", help="csv layer indices (default: model triple)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-free-hbm-gb", type=float, default=20.0)
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "data" / "issue_2569" / "xmodel"))
    ap.add_argument("--hf-data-repo", default="superkaiba1/explore-persona-space-data")
    ap.add_argument(
        "--hf-prefix",
        default=HF_XMODEL_PREFIX,
        help="destination prefix for finalized capture bundles",
    )
    ap.add_argument(
        "--qwen-gate",
        choices=("spot", "identity"),
        default="spot",
        help=(
            "Qwen correctness gate: banked same-text spot oracle (default), or the "
            "independent-boundary identity gate for alternate answer text"
        ),
    )
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--skip-gate-check",
        action="store_true",
        help="skip the gate-record precondition (smoke wiring only — never production)",
    )
    ap.add_argument("--gate-rel-tol", type=float, default=2e-2)
    ap.add_argument("--spot-chunk", default="shard00_chunk0000.pt")
    ap.add_argument("--pilot-booked-wall-h", type=float, default=6.0)
    ap.add_argument("--sentinel-path", default="")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: dispatch one phase; explicit flush + exit 0 (#1689 atexit rc)."""
    args = _parse_args(argv)
    if args.import_check:
        # Execute EVERY deferred (function-body) import on the real branch —
        # argcheck alone does not resolve them (#1689 false-pass class).
        import inspect as _inspect  # noqa: F401  (mirrors _load_model_ctx)

        import issue2476_turnavg_sae as _t24  # noqa: F401
        from huggingface_hub import HfApi as _HfApi  # noqa: F401
        from huggingface_hub.hf_api import RepoFile as _RepoFile  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForCausalLM as _AM,
            AutoTokenizer as _AT,
        )

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    assert args.phase, "--phase is required (or --import-check)"
    PHASES[args.phase](args)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
