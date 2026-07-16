#!/usr/bin/env python
"""Issue #1336 — Phase E: teacher-forced turn-store extraction (one cell/run).

One teacher-forced forward per conversation batch yields, per row:
  - slot vectors: residual activation at the prefix-header slot and the
    assistant-header slot, all 32 layers -> (2, 32, 4096)
  - span profiles: mean residual activation over the u1 and answer content
    spans, all 32 layers -> (2, 32, 4096)
  - per-turn mean teacher-forced NLL from the SAME logits
NO per-position capture (plan section 4 scope reduction #3).

Storage parity with the #825 parent (issue825_extract_turnstore.py:335-340):
bf16 compute AND bf16 shard store (fp16 overflows residual outlier dims);
finiteness asserted before storage. Shards of 500 rows, block-wise
extract->flush so host RAM holds ~one shard. Output stems are
``{model}_{format}_{corpus}`` so ``issue825_fit_cells._load_bundle_pt``
(track := corpus) loads them unchanged.

Runtime cross-model assert (plan section 8 risk row): the CONTEXT render of
100 sampled staged prompts must tokenize to IDENTICAL ids under every
checkpoint's tokenizer — each cell writes a context-token-id hash JSON and
asserts equality against any sibling model's hash for the same
(format, corpus).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch import

import torch  # noqa: E402

# Reused #825 helpers (arg-pure: no dependence on the parent's Qwen globals).
from issue825_extract_turnstore import (  # noqa: E402
    _finite,
    _git_commit,
    _ordered_slots,
    _ordered_turns,
    _turn_nll,
)
from issue1336_render import RENDERERS, validate_render  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

SHARD_SIZE = 500

# Architecture invariants; rebound from the tiny model's config in smoke mode
# (the parent --tiny-model-dir pattern: asserts stay ACTIVE, validating
# internal consistency instead of the 8B constants).
_EXPECTED_LAYERS = cm.EXPECTED_LAYERS
_EXPECTED_HIDDEN = cm.EXPECTED_HIDDEN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=tuple(cm.MODELS), required=True)
    parser.add_argument("--corpus", choices=tuple(cm.CORPORA), required=True)
    parser.add_argument("--format", choices=("chat", "naturalistic"), required=True)
    parser.add_argument("--gen-root", type=Path, default=None, help="generation outputs root")
    parser.add_argument("--prompts-root", type=Path, default=None, help="staged prompts root")
    parser.add_argument("--out-dir", type=Path, default=None, help="turnstore output dir")
    parser.add_argument("--batch-size", type=int, default=8, help="start size; halves on OOM")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--assert-causal", action="store_true", help="prefix-vs-full slot check")
    parser.add_argument("--smoke", action="store_true", help="smoke roots; causal check ON")
    parser.add_argument(
        "--tiny-model-dir",
        default=None,
        help=(
            "SMOKE ONLY: load a tiny random-init same-arch model (real tokenizer) "
            "from this dir; expected dims rebind to ITS config. Production runs "
            "NEVER pass this."
        ),
    )
    parser.add_argument("--ctx-hash-n", type=int, default=100, help="sampled contexts to hash")
    parser.add_argument("--upload", action="store_true", help="per-cell HF upload after extract")
    parser.add_argument(
        "--convention",
        choices=("committed", "corrected"),
        default="committed",
        help=(
            "capture convention (plan v7 Phase D2). 'committed' = current behavior "
            "(default; byte-identical when the D2 flags are absent). 'corrected' "
            "applies the slot/span offset override the D1.3 spot-check emitted "
            "(--offset-override + an explicit --out-dir REQUIRED)."
        ),
    )
    parser.add_argument(
        "--offset-override",
        type=Path,
        default=None,
        help=(
            "JSON the D1.3 spot-check emits when it indicts a specific offset: "
            '{"slot_offsets": {slot: int}, "span_offsets": {turn: [dstart, dend]}}. '
            "Consumed ONLY under --convention corrected (D2 needs no code change "
            "when it fires — the indicted offset is data, not code)."
        ),
    )
    parser.add_argument(
        "--row-allowlist",
        type=Path,
        default=None,
        help=(
            "JSON array of conv_ids ('s<prompt_idx>') or integer prompt_idx values; "
            "restricts extraction to exactly these kept rows (plan v7 D2: 512 "
            "wave-1 rows, same prompt ids). Every listed row must exist (fail-loud)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase D2 (conditional capture-parity probe) — convention + row subset
# ---------------------------------------------------------------------------
def resolve_convention(convention: str, offset_override: Path | None, out_dir: Path | None):
    """Validate the D2 flag combination; return the parsed override (or None).

    committed: no override may be passed (default behavior stays byte-identical).
    corrected: requires the D1.3-emitted override JSON AND an explicit --out-dir
    (corrected shards must never land in the committed turnstore dir).
    """
    if convention == "committed":
        assert offset_override is None, (
            "--offset-override is only consumed under --convention corrected"
        )
        return None
    assert offset_override is not None, (
        "--convention corrected requires --offset-override (the D1.3-emitted JSON)"
    )
    assert out_dir is not None, (
        "--convention corrected requires an explicit --out-dir — corrected shards "
        "must never overwrite the committed turnstore"
    )
    raw = json.loads(offset_override.read_text())
    slot_offsets = {str(k): int(v) for k, v in raw.get("slot_offsets", {}).items()}
    span_offsets = {str(k): (int(v[0]), int(v[1])) for k, v in raw.get("span_offsets", {}).items()}
    assert slot_offsets or span_offsets, (
        f"{offset_override}: override names no slot_offsets/span_offsets — the "
        "corrected convention is only meaningful when D1.3 indicted a specific offset"
    )
    return {"slot_offsets": slot_offsets, "span_offsets": span_offsets}


def apply_offset_override(r, override: dict):
    """Return a corrected copy of one Rendered row with shifted slots/spans.

    The corrected row is re-validated with the consumer-exact asserts — an
    override that produces an invalid render fails loud, never extracts.
    """

    slot_idx = dict(r.slot_idx)
    for name, dv in override["slot_offsets"].items():
        assert name in slot_idx, f"{r.conv_id}: slot_offsets names unknown slot {name!r}"
        slot_idx[name] = int(slot_idx[name]) + dv
    spans = dict(r.spans)
    for name, (ds, de) in override["span_offsets"].items():
        assert name in spans, f"{r.conv_id}: span_offsets names unknown span {name!r}"
        s, e = spans[name]
        spans[name] = (int(s) + ds, int(e) + de)
    corrected = dataclasses.replace(r, slot_idx=slot_idx, spans=spans)
    reason = validate_render(corrected)
    assert reason is None, (
        f"{corrected.conv_id}: corrected render invalid ({reason}) under the offset "
        "override — refusing to extract a convention that breaks the consumer asserts"
    )
    return corrected


def filter_row_allowlist(kept: list[dict], path: Path) -> list[dict]:
    """Restrict kept rows to the allowlist; every listed row must resolve."""
    entries = json.loads(path.read_text())
    assert isinstance(entries, list) and entries, f"{path}: allowlist must be a non-empty list"
    want = {e if isinstance(e, str) else f"s{int(e)}" for e in entries}
    picked = [r for r in kept if f"s{r['prompt_idx']}" in want]
    got = {f"s{r['prompt_idx']}" for r in picked}
    missing = sorted(want - got)
    assert not missing, (
        f"{path}: {len(missing)} allowlist rows not found among kept rows "
        f"(e.g. {missing[:5]}) — the allowlist must name existing wave-1 rows"
    )
    return picked


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode line iteration — never splitlines() (U+2028 in real user text)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model(slug: str, tiny_model_dir: str | None = None):
    """Load one ladder checkpoint (bf16, all weights pinned to GPU) or the tiny smoke model."""
    global _EXPECTED_LAYERS, _EXPECTED_HIDDEN
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tiny_model_dir is not None:
        model_id = f"TINY::{tiny_model_dir}"
        tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
        model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, torch_dtype=torch.float32)
        model.eval()
        _EXPECTED_LAYERS = int(model.config.num_hidden_layers)
        _EXPECTED_HIDDEN = int(model.config.hidden_size)
        return model, tokenizer, model_id

    model_id = cm.MODELS[slug]["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # device_map={"": 0} pins ALL weights to the GPU: a lingering engine
    # holding VRAM raises CUDA OOM at load instead of device_map="auto"
    # silently offloading layers to host RAM (parent runs 3-4, rc=137).
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    off_gpu = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    assert not off_gpu, (
        f"{model_id}: {len(off_gpu)} params not on CUDA (e.g. {off_gpu[:3]}) — "
        "refusing to run with CPU-offloaded weights (host-RAM OOM risk)"
    )
    cfg = model.config
    assert cfg.num_hidden_layers == _EXPECTED_LAYERS, (
        f"{model_id}: num_hidden_layers={cfg.num_hidden_layers} != {_EXPECTED_LAYERS}"
    )
    assert cfg.hidden_size == _EXPECTED_HIDDEN, (
        f"{model_id}: hidden_size={cfg.hidden_size} != {_EXPECTED_HIDDEN}"
    )
    return model, tokenizer, model_id


# ---------------------------------------------------------------------------
# Cross-model context-token-id hash (plan section 8: tokenizer/render parity)
# ---------------------------------------------------------------------------
def _context_text(fmt: str, question: str) -> str:
    """The CONTEXT (prefix + user query + assistant header) render per format."""
    if fmt == "chat":
        return cm.tulu_prompt(question)
    return f"User: {question}\n\nAssistant: "


def ctx_tokenid_hash(tokenizer, prompts: list[dict], fmt: str, n_sample: int) -> dict:
    """sha256 over the BOS-prepended context token ids of n_sample prompts."""
    import numpy as np

    idx = np.random.default_rng(0).choice(
        len(prompts), size=min(n_sample, len(prompts)), replace=False
    )
    idx = sorted(int(i) for i in idx)
    bos = tokenizer.bos_token_id
    assert bos is not None, "tokenizer has no BOS token"
    h = hashlib.sha256()
    for i in idx:
        ids = [
            int(bos),
            *tokenizer(_context_text(fmt, prompts[i]["prompt"]), add_special_tokens=False)[
                "input_ids"
            ],
        ]
        h.update((" ".join(map(str, ids)) + "\n").encode("utf-8"))
    return {"n_sampled": len(idx), "prompt_indices": idx, "sha256": h.hexdigest()}


def assert_ctx_hash_parity(out_dir: Path, slug: str, fmt: str, corpus: str, payload: dict) -> None:
    """Write this model's hash; assert equality with every sibling model's hash."""
    out_dir.mkdir(parents=True, exist_ok=True)
    own = out_dir / f"ctxhash_{fmt}_{corpus}_{slug}.json"
    own.write_text(json.dumps(payload, indent=2) + "\n")
    for other in cm.MODELS:
        if other == slug:
            continue
        sib = out_dir / f"ctxhash_{fmt}_{corpus}_{other}.json"
        if not sib.exists():
            continue
        sib_payload = json.loads(sib.read_text())
        assert sib_payload["sha256"] == payload["sha256"], (
            f"context token-id hash MISMATCH: {slug} vs {other} on ({fmt}, {corpus}) — "
            "the shared-render identical-ids assumption (plan section 4) is violated"
        )
        print(f"[ctxhash] parity OK vs {other} ({payload['sha256'][:12]}…)")


# ---------------------------------------------------------------------------
# Batched teacher-forced capture (NO per-position capture — plan section 4)
# ---------------------------------------------------------------------------
def process_batch(model, batch: list, pad_id: int, align_state: dict) -> list[dict]:
    """One forward per batch -> per-row slot vectors, span profiles, per-turn NLL."""
    lengths = [len(r.input_ids) for r in batch]
    bsz, max_len = len(batch), max(lengths)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, r in enumerate(batch):
        input_ids[i, : lengths[i]] = torch.tensor(r.input_ids, dtype=torch.long)
        attention_mask[i, : lengths[i]] = 1
    device = model.device
    captured, logits = extract_layer_activations(
        model,
        input_ids.to(device),
        layers=range(_EXPECTED_LAYERS),
        return_logits=True,  # logits ARE read (per-turn NLL) — no logits_to_keep
        attention_mask=attention_mask.to(device),
        # Keep activations ON DEVICE; only REDUCED tensors move to CPU (parent
        # round-3 review: never ship the full (L,B,T,H) grid over PCIe).
        detach_to_cpu=False,
    )
    assert set(captured) == set(range(_EXPECTED_LAYERS)), "missing layers in capture"
    acts = torch.stack([captured[layer] for layer in range(_EXPECTED_LAYERS)], dim=0)
    assert acts.shape == (_EXPECTED_LAYERS, bsz, max_len, _EXPECTED_HIDDEN), (
        f"acts shape {tuple(acts.shape)}"
    )
    records: list[dict] = []
    for i, r in enumerate(batch):
        true_len = lengths[i]
        slots = _ordered_slots(r)
        turns = _ordered_turns(r)
        for name, idx in slots:
            assert 0 <= idx < true_len, f"{r.conv_id}: slot {name}={idx} beyond len {true_len}"
        for name, (s, e) in turns:
            assert 1 <= s < e <= true_len, (
                f"{r.conv_id}: span {name}=({s},{e}) invalid for unpadded len {true_len}"
            )
        slot_pos = torch.tensor([idx for _, idx in slots], dtype=torch.long)
        slot_vecs = acts[:, i, slot_pos.to(acts.device), :].permute(1, 0, 2).contiguous().cpu()
        assert slot_vecs.shape == (len(slots), _EXPECTED_LAYERS, _EXPECTED_HIDDEN)
        profiles = torch.stack(
            [acts[:, i, s:e, :].float().mean(dim=1) for _, (s, e) in turns], dim=0
        ).cpu()
        assert profiles.shape == (len(turns), _EXPECTED_LAYERS, _EXPECTED_HIDDEN)
        nll = _turn_nll(logits[i], input_ids[i], true_len, turns, r.conv_id, align_state)
        assert nll.shape == (len(turns),)
        records.append(
            {
                "conv_id": r.conv_id,
                # bf16, NOT fp16: residual outlier dims can exceed fp16's 65504
                # max and silently become inf (parent code-review round-1;
                # issue825_extract_turnstore.py:335-340 parity).
                "slots": _finite(slot_vecs.to(torch.bfloat16), "slots", r.conv_id),
                "profiles": _finite(profiles.to(torch.bfloat16), "profiles", r.conv_id),
                "nll": _finite(nll, "nll", r.conv_id),
                "spans_meta": {
                    "conv_id": r.conv_id,
                    "format": r.format,
                    "seq_len": true_len,
                    "slot_names": [n for n, _ in slots],
                    "slot_idx": {n: int(v) for n, v in slots},
                    "turn_names": [n for n, _ in turns],
                    "spans": {n: [int(s), int(e)] for n, (s, e) in turns},
                    "meta": r.meta,
                },
            }
        )
    del captured, acts, logits
    return records


def causal_check(model, rendered: list, atol: float = 1e-2, n_conversations: int = 3) -> float:
    """Re-forward the prefix ending at each slot; slot activation must match full-seq.

    Inherited parent assert (plan section 4: causal-slot equality) with the
    layer count parametrized to this family's 32 layers.
    """
    device = model.device
    max_diff = 0.0
    n_checked = min(n_conversations, len(rendered))
    for r in rendered[:n_checked]:
        ids = torch.tensor(r.input_ids, dtype=torch.long).unsqueeze(0).to(device)
        full = extract_layer_activations(
            model, ids, layers=range(_EXPECTED_LAYERS), detach_to_cpu=True
        )
        for name, idx in _ordered_slots(r):
            pre = extract_layer_activations(
                model, ids[:, : idx + 1], layers=range(_EXPECTED_LAYERS), detach_to_cpu=True
            )
            for layer in range(_EXPECTED_LAYERS):
                a = pre[layer][0, idx].float()
                b = full[layer][0, idx].float()
                diff = float((a - b).abs().max())
                max_diff = max(max_diff, diff)
                assert torch.allclose(a, b, atol=atol), (
                    f"causal-slot mismatch {r.conv_id}:{name} layer {layer}: "
                    f"max|diff|={diff:.4g} > atol={atol}"
                )
    print(f"[causal] slot-prefix equality OK on {n_checked} rows; max|diff|={max_diff:.4g}")
    return max_diff


def run_extraction(model, rendered: list, pad_id: int, batch_size: int) -> list[dict]:
    """Length-grouped batching with OOM-halving (floor 1); restores input order."""
    order = sorted(range(len(rendered)), key=lambda j: len(rendered[j].input_ids))
    align_state: dict = {}
    results: dict[int, dict] = {}
    bs = batch_size
    pos = 0
    batches_done = 0
    while pos < len(order):
        chunk_idx = order[pos : pos + bs]
        chunk = [rendered[j] for j in chunk_idx]
        try:
            recs = process_batch(model, chunk, pad_id, align_state)
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            torch.cuda.empty_cache()
            print(f"[oom] CUDA OOM — halving batch size to {bs}")
            continue
        for j, rec in zip(chunk_idx, recs, strict=True):
            results[j] = rec
        pos += len(chunk_idx)
        batches_done += 1
        if batches_done % 10 == 0 or pos >= len(order):
            print(f"[extract] {pos}/{len(order)} rows done (batch size {bs})", flush=True)
    return [results[j] for j in range(len(rendered))]


def write_shards(
    records: list[dict],
    out_dir: Path,
    stem: str,
    sidecar_base: dict,
    shard_offset: int = 0,
    shard_size: int = SHARD_SIZE,
) -> list[Path]:
    """Write records as bf16 .pt shard(s) + JSON sidecars (parent contract, no perpos).

    ``issue825_fit_cells._load_bundle_pt`` tolerates the absent perpos keys
    (its key loop is presence-gated), so the fit loader is unchanged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for k in range(0, len(records), shard_size):
        shard = records[k : k + shard_size]
        shard_idx = shard_offset + k // shard_size
        payload = {
            "conv_ids": [r["conv_id"] for r in shard],
            "slots": [r["slots"] for r in shard],
            "profiles": [r["profiles"] for r in shard],
            "nll": [r["nll"] for r in shard],
            "spans_meta": [r["spans_meta"] for r in shard],
        }
        pt_path = out_dir / f"{stem}_shard{shard_idx:03d}.pt"
        torch.save(payload, pt_path)
        sidecar = dict(sidecar_base)
        sidecar.update(
            {
                "shard_index": shard_idx,
                "n_conversations": len(shard),
                "conv_ids": payload["conv_ids"],
                "shapes": {
                    "slots": [list(r["slots"].shape) for r in shard],
                    "profiles": [list(r["profiles"].shape) for r in shard],
                    "nll": [list(r["nll"].shape) for r in shard],
                },
            }
        )
        json_path = out_dir / f"{stem}_shard{shard_idx:03d}.json"
        json_path.write_text(json.dumps(sidecar, indent=2))
        paths.append(pt_path)
        print(f"[write] {pt_path} ({len(shard)} rows)")
    return paths


def _upload_cell(out_dir: Path, stem: str) -> None:
    """Per-cell incremental upload: ONE folder commit for this stem's files (#664).

    The ``{stem}.done.json`` marker is deliberately NOT uploaded: it is the
    LOCAL resume marker carrying the done == uploaded flag (its Hub copy
    would be stale-by-construction, written before the upload it records).
    """
    from huggingface_hub import upload_folder

    from explore_persona_space.orchestrate import hub

    prefix = f"{cm.HF_PREFIX_1336}/analysis_tensors/turnstore_{stem}"
    # Dir-filecount guard (#1190) OUTSIDE the retry wrapper (a guard raise is
    # deterministic; retrying it burns the budget for nothing).
    hub.assert_hub_dir_filecounts(out_dir, prefix, allow_patterns=[f"{stem}_shard*"])
    hub.retry_transient(
        lambda: upload_folder(
            repo_id=cm.HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(out_dir),
            path_in_repo=prefix,
            allow_patterns=[f"{stem}_shard*"],
            commit_message=f"issue-1336: turnstore {stem}",
        ),
        what=f"turnstore upload {stem}",
    )
    print(f"[upload] {stem} -> {prefix}")


def _write_done(done_path: Path, done: dict) -> None:
    """Atomic done-marker write (tmp + replace; the resume predicate reads it)."""
    tmp = done_path.with_suffix(done_path.suffix + ".tmp")
    tmp.write_text(json.dumps(done, indent=2) + "\n")
    tmp.replace(done_path)


def _hf_turnstore_listing(stem: str) -> list[str] | None:
    """File names under the cell's Hub turnstore prefix, or None when absent.

    Scoped ``list_repo_tree`` (never a full-repo listing — gotchas.md #833),
    MATERIALIZED inside the retry thunk (hub list APIs are lazy generators).
    A missing prefix (fresh cell) returns None; any other Hub failure raises
    through ``hub.retry_transient``.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

    from explore_persona_space.orchestrate import hub

    prefix = f"{cm.HF_PREFIX_1336}/analysis_tensors/turnstore_{stem}"
    api = HfApi()
    try:
        entries = hub.retry_transient(
            lambda: list(
                # HUB_VERIFY_RETRY_EXEMPT: scoped walk inside a retry_transient thunk
                api.list_repo_tree(
                    cm.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=False
                )
            ),
            what=f"turnstore HF-resume listing {stem}",
        )
    except EntryNotFoundError:
        return None
    except HfHubHTTPError as err:
        if getattr(getattr(err, "response", None), "status_code", None) == 404:
            return None
        raise
    return [Path(e.path).name for e in entries]


def _try_hf_resume(out_dir: Path, stem: str) -> dict | None:
    """Fetch a COMPLETE turnstore cell from HF into ``out_dir`` (resume path).

    Plan v9 route 1 resume: Phase E has cells already uploaded by the
    original run (done == uploaded, #664) — a fresh instance downloads them
    instead of re-extracting on GPU. Completeness is FAIL-LOUD: a partial
    Hub prefix (a .pt without its sidecar, a shard-index gap) raises rather
    than silently re-extracting or half-staging — a half-done cell must be
    triaged, never skipped past. Returns the done dict on success, None when
    the cell is not on the Hub at all.
    """
    import re

    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    names = _hf_turnstore_listing(stem)
    if not names:
        return None
    shard_re = re.compile(rf"^{re.escape(stem)}_shard(\d{{3}})\.(pt|json)$")
    shards: dict[int, set[str]] = {}
    for name in names:
        m = shard_re.match(name)
        if m:
            shards.setdefault(int(m.group(1)), set()).add(m.group(2))
    assert shards, (
        f"HF turnstore prefix for {stem} exists but holds no shard files ({sorted(names)[:8]}…) — "
        "partial/foreign upload; triage before resuming"
    )
    idxs = sorted(shards)
    ext_map = {i: sorted(e) for i, e in shards.items()}
    complete = idxs == list(range(len(idxs))) and all(shards[i] == {"pt", "json"} for i in idxs)
    assert complete, (
        f"HF turnstore for {stem} is INCOMPLETE (shard indices {idxs}, exts {ext_map}) — a "
        "half-done cell must be re-extracted or repaired explicitly, never silently resumed"
    )
    prefix = f"{cm.HF_PREFIX_1336}/analysis_tensors/turnstore_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for i in idxs:
        for ext in ("pt", "json"):
            rel = f"{prefix}/{stem}_shard{i:03d}.{ext}"
            local = hub.retry_transient(
                lambda r=rel: hf_hub_download(
                    repo_id=cm.HF_DATA_REPO, repo_type="dataset", filename=r, local_dir=out_dir
                ),
                what=f"turnstore HF-resume download {rel}",
            )
            # local_dir staging nests under the repo-relative path — move the
            # real file into the flat turnstore layout the fit loader reads.
            os.replace(local, out_dir / Path(rel).name)
            n_files += 1
    done = {
        "stem": stem,
        "n_rows": None,  # unknown without opening shards; sidecars carry it
        "n_shards": len(idxs),
        "uploaded": True,  # resumed FROM the Hub — done == uploaded holds
        "hf_resumed": True,
    }
    _write_done(out_dir / f"{stem}.done.json", done)
    print(f"[extract] {stem}: HF-resume fetched {n_files} files ({len(idxs)} shards) -> {out_dir}")
    return done


def main() -> None:
    args = parse_args()
    slug, fmt, corpus = args.model, args.format, args.corpus
    assert fmt in cm.FORMATS_BY_CORPUS[corpus], f"format {fmt} not registered for {corpus}"
    override = resolve_convention(args.convention, args.offset_override, args.out_dir)
    smoke = args.smoke
    data_root = Path("data/issue_1336")
    gen_root = args.gen_root or (data_root / ("gen_smoke" if smoke else "gen"))
    prompts_root = args.prompts_root or (data_root / ("prompts_smoke" if smoke else "prompts"))
    out_dir = args.out_dir or (data_root / ("turnstore_smoke" if smoke else "turnstore"))
    stem = cm.cell_id(slug, fmt, corpus)
    done_path = out_dir / f"{stem}.done.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if args.upload and not done.get("uploaded"):
            # done == uploaded (#664 per-cell contract): extraction completed
            # but the per-cell upload did not (transient Hub failure, or a
            # prior no-upload run) — re-attempt ONLY the upload, never the
            # extraction, and flip the flag only after it succeeds.
            print(f"[extract] {stem}: done marker exists, upload incomplete — re-uploading")
            _upload_cell(out_dir, stem)
            done["uploaded"] = True
            _write_done(done_path, done)
        print(f"[extract] skip {stem} (done marker exists)")
        return

    # Resume (plan v9 route 1): a cell the ORIGINAL run already extracted +
    # uploaded (done == uploaded, #664) is fetched from the Hub instead of
    # re-extracted on GPU. Smoke never touches the Hub (gen-script parity).
    if not smoke and _try_hf_resume(out_dir, stem) is not None:
        return

    rows = _read_jsonl(gen_root / slug / corpus / "answers.jsonl")
    kept = [r for r in rows if r.get("kept")]
    assert kept, f"no kept rows for {slug}/{corpus} under {gen_root}"
    if args.row_allowlist is not None:
        kept = filter_row_allowlist(kept, args.row_allowlist)
        print(f"[extract] row allowlist: {len(kept)} rows from {args.row_allowlist}")
    prompts = _read_jsonl(prompts_root / f"{corpus}.jsonl")
    assert prompts, f"no staged prompts for {corpus} under {prompts_root} — run gen --prep first"

    model, tokenizer, model_id = load_model(slug, tiny_model_dir=args.tiny_model_dir)

    # Runtime cross-model context-token-id parity assert (before any capture).
    hash_payload = ctx_tokenid_hash(tokenizer, prompts, fmt, args.ctx_hash_n)
    assert_ctx_hash_parity(out_dir, slug, fmt, corpus, hash_payload)

    rendered = []
    for r in kept:
        conv = {"conv_id": f"s{r['prompt_idx']}", "u1": r["prompt"], "a1": r["response"]}
        rr = RENDERERS[fmt](conv, tokenizer)
        reason = validate_render(rr)
        # Rows passed these EXACT asserts at gen time; a mismatch here is
        # tokenizer/render drift between phases — fail loud, never skip.
        assert reason is None, f"{rr.conv_id}: render invalid at extract time: {reason}"
        if override is not None:
            rr = apply_offset_override(rr, override)
        rendered.append(rr)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    causal_max_diff = causal_check(model, rendered) if (args.assert_causal or smoke) else None
    bs = max(1, int(args.batch_size))
    if smoke:
        bs = min(bs, 2)
    shard_size = int(args.shard_size)
    assert shard_size >= 1
    print(
        f"[run] cell={stem} model_id={model_id} n={len(rendered)} "
        f"batch_size={bs} layers={_EXPECTED_LAYERS} hidden={_EXPECTED_HIDDEN}",
        flush=True,
    )
    sidecar_base = {
        "model": slug,
        "model_id": model_id,
        "format": fmt,
        "corpus": corpus,
        "track": corpus,  # loader stem convention: {model}_{format}_{track}
        "expected_layers": _EXPECTED_LAYERS,
        "expected_hidden": _EXPECTED_HIDDEN,
        "shard_size": shard_size,
        "git_commit": _git_commit(),
        "args": {k: str(v) for k, v in vars(args).items()},
        "causal_check_max_abs_diff": causal_max_diff,
        "ctx_tokenid_sha256": hash_payload["sha256"],
        "smoke": bool(smoke),
        "convention": args.convention,
        "offset_override": override,
    }
    # Block-wise extract -> flush (parent run-4/run-5 RSS lessons): one block
    # == one shard file, written the moment its block completes.
    paths: list[Path] = []
    n_done = 0
    for block_idx, block_start in enumerate(range(0, len(rendered), shard_size)):
        block = rendered[block_start : block_start + shard_size]
        records = run_extraction(model, block, pad_id, bs)
        assert len(records) == len(block), (block_idx, len(records), len(block))
        paths += write_shards(
            records, out_dir, stem, sidecar_base, shard_offset=block_idx, shard_size=shard_size
        )
        n_done += len(records)
        del records, block
        gc.collect()
        # Return freed arena pages to the OS (parent run-5 monotone-RSS fix).
        with contextlib.suppress(OSError):
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    # Extraction state persists immediately (a crash in the upload below must
    # not forfeit the GPU work), but with uploaded=False: the resume predicate
    # above re-attempts the upload — done == uploaded (#664), and a transient
    # Hub failure is retried on the next run instead of silently skipped.
    done = {"stem": stem, "n_rows": n_done, "n_shards": len(paths), "uploaded": False}
    _write_done(done_path, done)
    if args.upload:
        _upload_cell(out_dir, stem)
        done["uploaded"] = True
        _write_done(done_path, done)
    print(f"[done-cell] {stem}: {n_done} rows -> {len(paths)} shard(s) in {out_dir}")


if __name__ == "__main__":
    main()
