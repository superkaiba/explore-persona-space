#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #920 S3: extract ALL 55 per-layer summary families (GPU, HF, own process).

One teacher-forced forward per (context, probe) row — ``prompt + stored answer +
the 5-token boundary block <|im_end|>\\n<|im_start|>user\\n`` — computes EVERY
per-layer family (context 19 + answer 16 + positions 20, plan §3.1–3.3) GPU-side
IN-FORWARD in fp32, with ONE PCIe transfer per left-padded batch. Extends the
#810 ``_run_forward_batch`` pattern (left-pad + explicit position_ids, GPU-side
gathers) with exact content masks from the equality-asserted chat-template
reconstruction (``issue920_common.build_prompt_ids_with_masks``).

Per-probe persistence (the pinned ``probe_avg_max`` convention: layer-pooled +
probe-mean reductions happen at FIT time from the per-probe store): one
``<ctx>.pt`` per context per probe set, ``{family: (n_probes, Lc, H)}`` fp16
(means / singles / positions) / bf16 (max families) + a (n_probes, 55) validity
mask, with the fail-loud ``|x| < 6e4`` assert on every fp16-bound tensor.

The §7 G1 equivalence gate (``--equiv-gate-first``) runs BEFORE the full
extraction: 2–3 contexts vs the #658 position store (probe-mean im_end /
turn_nl / head / tail, fp16 tolerance, #810-compat in-range reduction), a
context-side ``sigma_c.pt`` key-schema attempt with the batch-1-vs-batched
self-equivalence (cosine ≥ 0.999) fallback that ALWAYS runs.

Usage::

    # production (inside issue920_dispatch.sh; GPU):
    uv run python scripts/issue920_extract_summaries.py --gpu --equiv-gate-first \\
        --probe-set both --batch-probes 8

    # CPU smoke (tiny same-family model, real tokenizer/template, full path):
    uv run python scripts/issue920_extract_summaries.py --smoke \\
        --model Qwen/Qwen2.5-0.5B-Instruct --n-ctx 2 --n-probes 2 \\
        --probe-set A --out-root /tmp/i920_smoke_extract --no-upload
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Shared-VM thread caps (#847): dotenv before torch's import-time pool freeze.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))


import torch  # noqa: E402
from issue594_extract_context_vectors import LayerCapture  # noqa: E402
from issue920_common import (  # noqa: E402
    ALL_STORE_FAMILIES,
    DEFAULT_MODEL,
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    FP16_ABS_MAX,
    HF_DATA_REPO,
    I658_G1_SIGMA_C,
    I658_POSITION_STORE_PREFIX,
    I658_RAW_COMPLETIONS_PREFIX,
    I920_SUMMARIES_PREFIX,
    I920_TENSORS_PREFIX,
    N_LASTK,
    PROBES_A_PATH,
    PROBES_B_PATH,
    VALID_MISSING,
    VALID_OK,
    assert_token_pins,
    build_full_row,
    dump_json,
    load_battery,
    load_json,
    load_probes,
    position_slots,
    reproducibility_metadata,
    resolve_hf_revision,
    store_dtype,
    write_sentinel,
)

logger = logging.getLogger("issue920_extract")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── family → reduction spec (masks + singles) ────────────────────────────────

# 8 token masks, each feeding a mean family and a max family (16 families).
MASK_NAMES = [
    "ctx_wt",
    "ctx_co",
    "ctx_blk",
    "ans_content",
    "ans_uhdr",
    "ans_blk5",
    "ans_wtn",
    "ans_wtf",
]
MASK_FAMILY = {  # family -> (mask index, "mean"|"max")
    "ctx_wt_mean": (0, "mean"),
    "ctx_wt_max": (0, "max"),
    "ctx_co_mean": (1, "mean"),
    "ctx_co_max": (1, "max"),
    "ctx_blk_mean": (2, "mean"),
    "ctx_blk_max": (2, "max"),
    "ans_content_mean": (3, "mean"),
    "ans_content_max": (3, "max"),
    "ans_uhdr_mean": (4, "mean"),
    "ans_uhdr_max": (4, "max"),
    "ans_blk5_mean": (5, "mean"),
    "ans_blk5_max": (5, "max"),
    "ans_wtn_mean": (6, "mean"),
    "ans_wtn_max": (6, "max"),
    "ans_wtf_mean": (7, "mean"),
    "ans_wtf_max": (7, "max"),
}
SINGLE_FAMILIES = [f for f in ALL_STORE_FAMILIES if f not in MASK_FAMILY]
assert len(SINGLE_FAMILIES) == 39
SINGLE_INDEX = {f: i for i, f in enumerate(SINGLE_FAMILIES)}
FAMILY_INDEX = {f: i for i, f in enumerate(ALL_STORE_FAMILIES)}


def _row_geometry(row: dict) -> tuple[dict[str, list[int]], list[int | None], list[int]]:
    """Mask token-index lists + single-position list (39) + validity (55) for one row.

    All indices are PRE-PAD absolute positions into ``row['full_ids']``; the batch
    flush shifts them by the left-pad amount. Validity is in the canonical
    ALL_STORE_FAMILIES order (mask families always VALID_OK; lastk/positions per
    their range/dedup rules).
    """
    p, a = row["prompt_len"], row["ans_len"]
    masks = {
        "ctx_wt": list(range(0, p)),
        "ctx_co": row["content_pos"],
        "ctx_blk": list(range(p - 5, p)),
        "ans_content": list(range(row["ans_start"], row["ans_start"] + a)),
        "ans_uhdr": [row["b_im_start"], row["b_user"], row["b_uh_nl"]],
        "ans_blk5": [
            row["b_im_end"],
            row["b_nl"],
            row["b_im_start"],
            row["b_user"],
            row["b_uh_nl"],
        ],
        "ans_wtn": list(range(row["ans_start"], row["b_nl"] + 1)),
        "ans_wtf": list(range(row["ans_start"], row["b_uh_nl"] + 1)),
    }
    assert masks["ctx_co"], "empty context content mask (probe content should be non-empty)"
    singles: list[int | None] = [
        row["ah_nl"],
        row["tt_im_end"],
        row["tt_nl"],
        row["tt_im_start"],
        row["tt_assistant"],
    ]
    singles += row["lastk_pos"]  # 8 entries, None when < k content tokens
    singles += [
        row["b_im_end"],
        row["ans_start"] + a - 1,
        row["b_nl"],
        row["b_im_start"],
        row["b_user"],
        row["b_uh_nl"],
    ]
    pos_rel, pos_valid = position_slots(a)
    singles += [None if r is None else row["ans_start"] + r for r in pos_rel]
    assert len(singles) == 39, len(singles)

    validity = []
    for fam in ALL_STORE_FAMILIES:
        if fam in MASK_FAMILY:
            validity.append(VALID_OK)
        elif fam.startswith("ctx_lastk_"):
            k = int(fam.rsplit("_", 1)[1])
            validity.append(VALID_OK if row["lastk_pos"][k - 1] is not None else VALID_MISSING)
        elif fam.startswith("pos_"):
            validity.append(pos_valid[POS_OFFSET[fam]])
        else:
            validity.append(VALID_OK)
    return masks, singles, validity


POS_OFFSET = {
    f: i
    for i, f in enumerate(
        [f"pos_head_{j}" for j in range(10)] + [f"pos_tail_{k}" for k in range(1, 11)]
    )
}

# sanity: SINGLE_FAMILIES order must match the singles list built in _row_geometry
_EXPECTED_SINGLE_ORDER = (
    ["ctx_ah_nl", "ctx_tt_im_end", "ctx_tt_nl", "ctx_tt_im_start", "ctx_tt_assistant"]
    + [f"ctx_lastk_{k}" for k in range(1, N_LASTK + 1)]
    + [
        "ans_im_end",
        "ans_last_content",
        "ans_turn_nl",
        "ans_uh_im_start",
        "ans_uh_user",
        "ans_uh_nl",
    ]
    + list(POS_OFFSET)
)
assert SINGLE_FAMILIES == _EXPECTED_SINGLE_ORDER, "single-family order drift"


# ── batched forward + in-forward fp32 reductions ─────────────────────────────


def run_forward_batch(
    model, capture: LayerCapture, tokenizer, rows: list[dict], capture_layers: list[int]
) -> torch.Tensor:
    """Left-pad + ONE forward + GPU-side fp32 reductions for ALL 55 families.

    Returns (B, 55, Lc, H) fp32 on CPU (ONE transfer per batch). Threads explicit
    ``position_ids`` (cumsum(mask)−1 clamped at 0) — left-pad silently diverges
    from batch-1 without it (the #502 lesson). Invalid singles gather index 0 as
    a placeholder; callers key on the validity mask, never the placeholder.
    """
    device = model.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151645
    b = len(rows)
    lc = len(capture_layers)
    H = model.config.hidden_size
    max_len = max(len(r["full_ids"]) for r in rows)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((b, max_len), dtype=torch.long)
    mask_t = torch.zeros((len(MASK_NAMES), b, max_len), dtype=torch.bool)
    sing_t = torch.full((b, 39), -1, dtype=torch.long)
    for bi, r in enumerate(rows):
        L = len(r["full_ids"])
        pad = max_len - L  # LEFT-pad: real tokens occupy [pad, max_len)
        input_ids[bi, pad:] = torch.tensor(r["full_ids"], dtype=torch.long)
        attn[bi, pad:] = 1
        masks, singles, _validity = r["geometry"]
        for mi, mn in enumerate(MASK_NAMES):
            idx = torch.tensor(masks[mn], dtype=torch.long) + pad
            mask_t[mi, bi, idx] = True
        for si, pos in enumerate(singles):
            if pos is not None:
                sing_t[bi, si] = pos + pad
    input_ids = input_ids.to(device)
    attn = attn.to(device)
    position_ids = (attn.long().cumsum(dim=1) - 1).clamp(min=0).to(device)
    mask_dev = mask_t.to(device)
    sing_dev = sing_t.clamp(min=0).to(device)
    counts = mask_dev.sum(dim=2).clamp(min=1).to(torch.float32)  # (M, B)

    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids)

    out = torch.zeros((b, len(ALL_STORE_FAMILIES), lc, H), dtype=torch.float32, device=device)
    gidx = sing_dev.unsqueeze(-1).expand(b, 39, H)
    for li_out, li in enumerate(capture_layers):
        hs32 = capture.latest[li].to(torch.float32)  # (B, T, H)
        # masked means, all 8 masks in one einsum
        sums = torch.einsum("mbt,bth->mbh", mask_dev.to(torch.float32), hs32)
        means = sums / counts.unsqueeze(-1)  # (M, B, H)
        # masked per-dim maxes (loop the 8 masks; each is one masked_fill+amax)
        maxes = []
        for mi in range(len(MASK_NAMES)):
            filled = hs32.masked_fill(~mask_dev[mi].unsqueeze(-1), float("-inf"))
            maxes.append(filled.amax(dim=1))  # (B, H)
        picked = torch.gather(hs32, 1, gidx)  # (B, 39, H)
        for fam, fi in FAMILY_INDEX.items():
            if fam in MASK_FAMILY:
                mi, kind = MASK_FAMILY[fam]
                out[:, fi, li_out] = means[mi] if kind == "mean" else maxes[mi]
            else:
                out[:, fi, li_out] = picked[:, SINGLE_INDEX[fam]]
    capture.latest.clear()
    return out.cpu()  # ONE PCIe transfer per batch


def extract_context(
    model,
    tokenizer,
    capture,
    instance: dict,
    probes: list[str],
    completions: list[str],
    capture_layers: list[int],
    batch_probes: int,
    user_id: int,
) -> dict:
    """Per-probe store blob for one context: {family: (n_probes, Lc, H)} + validity."""
    rows = []
    kept_probes = []
    ans_lens = []
    empty = 0
    for q, ans in zip(probes, completions, strict=True):
        row = build_full_row(tokenizer, instance, q, ans, user_id)
        if row is None:
            empty += 1
            logger.warning("empty completion for %s probe=%r — skipping", instance["id"], q[:40])
            continue
        row["geometry"] = _row_geometry(row)
        rows.append(row)
        kept_probes.append(q)
        ans_lens.append(row["ans_len"])
    if not rows:
        raise RuntimeError(f"context {instance['id']}: every probe produced an empty answer")

    n = len(rows)
    lc = len(capture_layers)
    H = model.config.hidden_size
    vals = torch.zeros((n, len(ALL_STORE_FAMILIES), lc, H), dtype=torch.float32)
    validity = torch.zeros((n, len(ALL_STORE_FAMILIES)), dtype=torch.uint8)
    for bi, r in enumerate(rows):
        validity[bi] = torch.tensor(r["geometry"][2], dtype=torch.uint8)
    bs = max(1, int(batch_probes))
    for lo in range(0, n, bs):
        chunk = rows[lo : lo + bs]
        vals[lo : lo + len(chunk)] = run_forward_batch(
            model, capture, tokenizer, chunk, capture_layers
        )

    blob: dict = {
        "context_id": instance["id"],
        "families": ALL_STORE_FAMILIES,
        "capture_layers": capture_layers,
        "probes": kept_probes,
        "ans_lens": ans_lens,
        "validity": validity,  # (n_probes, 55) uint8: 0 missing / 1 ok / 2 dedup-masked
        "empty_completions": empty,
        "model": model.config._name_or_path,
    }
    for fam, fi in FAMILY_INDEX.items():
        t = vals[:, fi]  # (n, Lc, H) fp32
        dt = store_dtype(fam)
        if dt == torch.float16:
            amax = t.abs().amax().item()
            if amax >= FP16_ABS_MAX:
                raise RuntimeError(
                    f"[fp16-range-assert] family {fam} of {instance['id']} has |x| max "
                    f"{amax:.3e} ≥ {FP16_ABS_MAX:.0e} — fp16 persist would overflow"
                )
        blob[f"fam::{fam}"] = t.to(dt)
    return blob


# ── inputs ────────────────────────────────────────────────────────────────────


def load_completions(ctx_id: str, probe_set: str, gen_b_dir: Path) -> list[dict]:
    """Stored (probe, completion) rows: set A from the #658 HF bucket, set B local/HF."""
    if probe_set == "A":
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            HF_DATA_REPO, f"{I658_RAW_COMPLETIONS_PREFIX}/{ctx_id}.json", repo_type="dataset"
        )
        blob = load_json(path)
    else:
        local = gen_b_dir / f"{ctx_id}.json"
        if local.is_file():
            blob = load_json(local)
        else:
            from huggingface_hub import hf_hub_download
            from issue920_common import I920_GEN_B_PREFIX

            path = hf_hub_download(
                HF_DATA_REPO, f"{I920_GEN_B_PREFIX}/{ctx_id}.json", repo_type="dataset"
            )
            blob = load_json(path)
    if blob.get("context_id") != ctx_id:
        raise RuntimeError(f"completions ctx mismatch: {blob.get('context_id')} != {ctx_id}")
    cells = blob["completions"]
    if not cells:
        raise RuntimeError(f"context {ctx_id}: no stored completions")
    return cells


def _load_model(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map={"": torch.device("cuda:0")}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


# ── G1 equivalence gate ───────────────────────────────────────────────────────

_G1_POS_MAP = {  # #658 store position name -> our family
    "im_end": "ans_im_end",
    "turn_nl": "ans_turn_nl",
    **{f"tail_{k}": f"pos_tail_{k}" for k in range(1, 11)},
    **{f"head_{j}": f"pos_head_{j}" for j in range(10)},
}


def _in_range_probe_mean(blob: dict, family: str) -> tuple[torch.Tensor, int]:
    """#810-compat probe-mean: reduce over validity ∈ {OK, DEDUP} (in-range only)."""
    v = blob["validity"][:, FAMILY_INDEX[family]]
    keep = v > 0
    n = int(keep.sum())
    t = blob[f"fam::{family}"].to(torch.float32)[keep]
    return t.mean(dim=0), n


def _vec_close(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """(cosine, relative RMS error) between two flattened summary stacks."""
    af, bf = a.flatten().double(), b.flatten().double()
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    rel = (torch.linalg.norm(af - bf) / (torch.linalg.norm(bf) + 1e-9)).item()
    return cos, rel


def g1_gate(blobs: dict[str, dict], assert_pass: bool) -> dict:
    """Compare probe-mean position vectors vs the #658 position store (fp16 tol).

    ``assert_pass=False`` (smoke dry mode on a tiny model) computes + reports the
    diffs without asserting — the code path is exercised end-to-end; the real
    gate asserts cos ≥ 0.999 AND relative RMS ≤ 0.02 per position (the store's
    own fp16 precision), plus coverage equality.
    """
    from huggingface_hub import hf_hub_download

    report: dict = {"contexts": {}, "assert_pass": assert_pass}
    worst = (1.0, 0.0)
    for ctx_id, blob in blobs.items():
        path = hf_hub_download(
            HF_DATA_REPO, f"{I658_POSITION_STORE_PREFIX}/{ctx_id}.pt", repo_type="dataset"
        )
        store = torch.load(path, weights_only=False)
        pos_names = store["positions"]
        rows = {}
        for sname, fam in _G1_POS_MAP.items():
            if sname not in pos_names:
                continue
            ours, n_ours = _in_range_probe_mean(blob, fam)
            theirs = store["pos_vectors"][pos_names.index(sname)].to(torch.float32)
            n_theirs = int(store["coverage"].get(sname, 0))
            if ours.shape != theirs.shape:
                if assert_pass:
                    raise RuntimeError(
                        f"[g1-equiv-assert] {ctx_id}/{sname}: shape {tuple(ours.shape)} vs "
                        f"store {tuple(theirs.shape)} — wrong model/layer config"
                    )
                rows[sname] = {
                    "shape_mismatch": [list(ours.shape), list(theirs.shape)],
                    "note": "expected on the tiny-model dry smoke",
                }
                continue
            cos, rel = _vec_close(ours, theirs)
            rows[sname] = {
                "cos": round(cos, 6),
                "rel_rms": round(rel, 6),
                "coverage_ours": n_ours,
                "coverage_store": n_theirs,
            }
            worst = (min(worst[0], cos), max(worst[1], rel))
            if assert_pass:
                assert n_ours == n_theirs, (ctx_id, sname, n_ours, n_theirs)
                assert cos >= 0.999 and rel <= 0.02, (
                    f"[g1-equiv-assert] {ctx_id}/{sname}: cos={cos:.5f} rel_rms={rel:.5f} — "
                    "extraction drift vs the #658 position store"
                )
        report["contexts"][ctx_id] = rows
    report["worst_cos"], report["worst_rel_rms"] = worst
    logger.info(
        "[g1] position-store check: worst cos=%.5f worst rel_rms=%.5f (assert=%s)",
        worst[0],
        worst[1],
        assert_pass,
    )
    return report


def g1_context_side(blobs: dict[str, dict], assert_pass: bool) -> dict:
    """Context-side G1: try the #658 sigma_c.pt key schema; report or fall back.

    Assumption 9 (LOW confidence) — on ANY schema mismatch this logs + returns a
    ``fallback: self-equivalence`` verdict (the batch-1-vs-batched check below is
    the binding fallback, run unconditionally by the caller).
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(HF_DATA_REPO, I658_G1_SIGMA_C, repo_type="dataset")
        sig = torch.load(path, weights_only=False)
    except Exception as e:
        logger.warning("[g1] sigma_c.pt unavailable (%s) — falling back to self-equivalence", e)
        return {"mode": "fallback_self_equivalence", "reason": str(e)[:200]}
    rows = {}
    for ctx_id, blob in blobs.items():
        tensor = None
        if isinstance(sig, dict):
            entry = sig.get(ctx_id)
            if isinstance(entry, torch.Tensor):
                tensor = entry
            elif isinstance(entry, dict):
                for key in ("cc_last", "ah_nl", "last", "tensor"):
                    if isinstance(entry.get(key), torch.Tensor):
                        tensor = entry[key]
                        break
        if tensor is None or tensor.shape != blob["fam::ctx_ah_nl"].shape[1:]:
            logger.warning("[g1] sigma_c.pt key schema mismatch for %s — fallback", ctx_id)
            return {
                "mode": "fallback_self_equivalence",
                "reason": f"key schema mismatch at {ctx_id}",
            }
        ours, _n = _in_range_probe_mean(blob, "ctx_ah_nl")
        cos, rel = _vec_close(ours, tensor.to(torch.float32))
        rows[ctx_id] = {"cos": round(cos, 6), "rel_rms": round(rel, 6)}
        if assert_pass:
            assert cos >= 0.999, f"[g1-ctx-assert] {ctx_id}: ah_nl cos={cos:.5f} vs sigma_c"
    return {"mode": "sigma_c", "contexts": rows}


def g1_self_equivalence(
    model, tokenizer, capture, instance, probes, completions, capture_layers, user_id
) -> dict:
    """Batch-1 vs batched (left-pad fires) equivalence: cosine ≥ 0.999 per family.

    Runs the SAME production ``extract_context`` at batch len(probes) and batch 1
    and compares the per-probe fp32 values on three representative families
    (ah_nl single, content mean, head_0 position) — the batched-rewrite
    equivalence requirement (B≥2 so padding + position_ids are exercised).
    """
    batched = extract_context(
        model,
        tokenizer,
        capture,
        instance,
        probes,
        completions,
        capture_layers,
        batch_probes=len(probes),
        user_id=user_id,
    )
    serial = extract_context(
        model,
        tokenizer,
        capture,
        instance,
        probes,
        completions,
        capture_layers,
        batch_probes=1,
        user_id=user_id,
    )
    out = {}
    for fam in ("ctx_ah_nl", "ans_content_mean", "pos_head_0", "ctx_co_mean", "ans_wtf_max"):
        a = batched[f"fam::{fam}"].to(torch.float32)
        b = serial[f"fam::{fam}"].to(torch.float32)
        cos, rel = _vec_close(a, b)
        out[fam] = {"cos": round(cos, 6), "rel_rms": round(rel, 6)}
        assert cos >= 0.999, f"[g1-selfequiv-assert] {fam}: batched vs batch-1 cos={cos:.5f}"
    logger.info("[g1] batch-1 vs batched self-equivalence PASS: %s", out)
    return out


# ── upload ────────────────────────────────────────────────────────────────────


def upload_set(out_dir: Path, ctx_ids: list[str], probe_set: str) -> str:
    from huggingface_hub import HfApi, list_repo_files

    path_in_repo = I920_SUMMARIES_PREFIX[probe_set]
    api = HfApi()
    api.upload_folder(
        folder_path=str(out_dir),
        path_in_repo=path_in_repo,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=["*.pt", "manifest.json"],
        commit_message=f"issue #920: per-probe summary store set {probe_set} "
        f"({len(ctx_ids)} contexts)",
    )
    remote = set(list_repo_files(HF_DATA_REPO, repo_type="dataset", revision="main"))
    missing = {f"{path_in_repo}/{c}.pt" for c in ctx_ids} - remote
    if missing:
        raise RuntimeError(
            f"summary-store upload verification FAILED ({probe_set}): "
            f"{len(missing)} missing, e.g. {sorted(missing)[:3]}"
        )
    logger.info("summary store %s verified: %d contexts", probe_set, len(ctx_ids))
    return path_in_repo


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #920 S3: 55-family summary extraction")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--probe-set", choices=["A", "B", "both"], default="both")
    ap.add_argument("--out-root", default=str(PROJECT_ROOT / "data" / "issue_920"))
    ap.add_argument("--gen-b-dir", default=str(PROJECT_ROOT / "data" / "issue_920" / "gen_b"))
    ap.add_argument("--n-ctx", type=int, default=None)
    ap.add_argument("--n-probes", type=int, default=None)
    ap.add_argument(
        "--batch-probes",
        type=int,
        default=8,
        help="probes per left-padded forward (#810 measured 8; raise ≤16 only "
        "after a live mem_get_info read)",
    )
    ap.add_argument(
        "--equiv-gate-first",
        action="store_true",
        help="run the §7 G1 gate on --equiv-n-ctx contexts BEFORE extraction",
    )
    ap.add_argument("--equiv-n-ctx", type=int, default=2)
    ap.add_argument(
        "--equiv-gate-dry",
        action="store_true",
        help="run the G1 code path WITHOUT asserting (tiny-model smoke)",
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    device = args.device or ("cuda" if (args.gpu and torch.cuda.is_available()) else "cpu")
    t0 = time.time()
    out_root = Path(args.out_root)

    logger.info("[phase=setup] battery + probes")
    instances, _fam = load_battery()
    if args.n_ctx is not None:
        instances = instances[: args.n_ctx]
    ctx_ids = [i["id"] for i in instances]
    inst_by_id = {i["id"]: i for i in instances}
    sets = ["A", "B"] if args.probe_set == "both" else [args.probe_set]
    probes_by_set = {}
    for s in sets:
        probes_by_set[s] = [
            p["text"] for p in load_probes(PROBES_A_PATH if s == "A" else PROBES_B_PATH)
        ]

    logger.info("[phase=load_model] %s (device=%s)", args.model, device)
    model, tokenizer = _load_model(args.model, device)
    user_id, _assistant_id = assert_token_pins(tokenizer)
    n_layers = model.config.num_hidden_layers
    capture_layers = list(range(n_layers))
    if not args.smoke:
        assert n_layers == EXPECTED_LAYERS, (n_layers, EXPECTED_LAYERS)
        assert model.config.hidden_size == EXPECTED_HIDDEN, model.config.hidden_size
    capture = LayerCapture(model, n_layers)
    hf_revision = resolve_hf_revision() if not args.smoke else None

    gate_report = None
    try:
        # ── G1 gate FIRST (2–3 contexts, set A, all probes) ──────────────────
        if args.equiv_gate_first:
            assert_pass = not args.equiv_gate_dry
            logger.info("[phase=g1_gate] %d contexts (assert=%s)", args.equiv_n_ctx, assert_pass)
            gate_blobs = {}
            g1_ids = ctx_ids[: args.equiv_n_ctx]
            for cid in g1_ids:
                cells = load_completions(cid, "A", Path(args.gen_b_dir))
                if args.n_probes is not None:
                    cells = cells[: args.n_probes]
                gate_blobs[cid] = extract_context(
                    model,
                    tokenizer,
                    capture,
                    inst_by_id[cid],
                    [c["probe"] for c in cells],
                    [c["completion"] for c in cells],
                    capture_layers,
                    args.batch_probes,
                    user_id,
                )
            pos_report = g1_gate(gate_blobs, assert_pass=assert_pass)
            ctx_report = g1_context_side(gate_blobs, assert_pass=assert_pass)
            first = g1_ids[0]
            cells = load_completions(first, "A", Path(args.gen_b_dir))[: max(2, args.n_probes or 4)]
            self_report = g1_self_equivalence(
                model,
                tokenizer,
                capture,
                inst_by_id[first],
                [c["probe"] for c in cells],
                [c["completion"] for c in cells],
                capture_layers,
                user_id,
            )
            gate_report = {
                "positions": pos_report,
                "context_side": ctx_report,
                "self_equivalence": self_report,
            }
            dump_json(gate_report, out_root / "g1_gate_report.json")
            if not args.no_upload and not args.smoke:
                # Gate evidence is durable BEFORE the auto-delete lane exits
                # (single small file — the per-file API is correct here).
                from huggingface_hub import HfApi

                HfApi().upload_file(
                    path_or_fileobj=str(out_root / "g1_gate_report.json"),
                    path_in_repo=f"{I920_TENSORS_PREFIX}/g1_gate_report.json",
                    repo_id=HF_DATA_REPO,
                    repo_type="dataset",
                    commit_message="issue #920: G1 equivalence-gate report",
                )
                logger.info("g1_gate_report.json uploaded to %s/", I920_TENSORS_PREFIX)
            write_sentinel(
                "epm:progress",
                {
                    "phase": "S3_g1_gate",
                    "blocks_pipeline": False,
                    "gate": "PASS" if assert_pass else "DRY",
                    "worst_cos": pos_report["worst_cos"],
                },
                out_root,
                slug_extra="g1",
            )

        # ── full extraction per set (resume: skip existing per-context files) ─
        for s in sets:
            out_dir = out_root / f"summaries_set{s}"
            out_dir.mkdir(parents=True, exist_ok=True)
            probes = probes_by_set[s]
            if args.n_probes is not None:
                probes = probes[: args.n_probes]
            for ci, cid in enumerate(ctx_ids):
                target = out_dir / f"{cid}.pt"
                if target.is_file():
                    logger.info(
                        "[phase=extract_%s] %d/%d %s — exists, skip (resume)",
                        s,
                        ci + 1,
                        len(ctx_ids),
                        cid,
                    )
                    continue
                logger.info("[phase=extract_%s] %d/%d %s", s, ci + 1, len(ctx_ids), cid)
                cells = load_completions(cid, s, Path(args.gen_b_dir))
                if args.n_probes is not None:
                    cells = cells[: args.n_probes]
                stored_probes = [c["probe"] for c in cells]
                if s == "A" and args.n_probes is None:
                    assert stored_probes == probes_by_set["A"] or set(stored_probes) == set(
                        probes_by_set["A"]
                    ), f"set-A probe drift for {cid}"
                blob = extract_context(
                    model,
                    tokenizer,
                    capture,
                    inst_by_id[cid],
                    stored_probes,
                    [c["completion"] for c in cells],
                    capture_layers,
                    args.batch_probes,
                    user_id,
                )
                blob["probe_set"] = s
                torch.save(blob, target)
            manifest = {
                "probe_set": s,
                "families": ALL_STORE_FAMILIES,
                "n_contexts": len(ctx_ids),
                "context_ids": ctx_ids,
                "capture_layers": capture_layers,
                "dtype_split": {
                    "bf16": sorted(
                        f for f in ALL_STORE_FAMILIES if store_dtype(f) == torch.bfloat16
                    ),
                    "fp16": "all others",
                },
                "probe_avg_max_convention": (
                    "token-max per probe fp32 in-forward; layer pools + probe-mean derived "
                    "PER PROBE at fit time (max never averaged before the per-probe pool)"
                ),
                "validity_codes": {
                    "0": "missing (out of range)",
                    "1": "valid",
                    "2": "tail slot deduped by absolute position (≤ head "
                    "window); in #810-compat reductions, masked in fits",
                },
                "boundary_block": "[151645, 198, 151644, 872(runtime-asserted), 198]",
                "content_mask": (
                    "instance-provided system+user text tokens incl. the probe "
                    "turn; template tokens, template-injected default system, "
                    "and assistant prefix-turn content EXCLUDED"
                ),
                "hf_data_repo_revision_at_fetch": hf_revision,
                "model": args.model,
                "smoke": args.smoke,
                "reproducibility": reproducibility_metadata(),
            }
            dump_json(manifest, out_dir / "manifest.json")
            if not args.no_upload and not args.smoke:
                logger.info("[phase=upload_%s] summary store", s)
                upload_set(out_dir, ctx_ids, s)
            write_sentinel(
                "epm:progress",
                {
                    "phase": f"S3_extract_set{s}",
                    "blocks_pipeline": False,
                    "n_contexts": len(ctx_ids),
                },
                out_root,
                slug_extra=f"extract-{s}",
            )
    finally:
        capture.remove()

    # Post-upload phase-done marker: the dispatcher's resume predicate keys on
    # this, so a crash at a per-set upload re-enters the phase on retry (same
    # class as the post-K3 fit-done marker; internal per-context resume makes
    # the re-entry cheap).
    dump_json(
        {
            "phase": "S3_extract",
            "sets": sets,
            "n_contexts": len(ctx_ids),
            "reproducibility": reproducibility_metadata(),
        },
        out_root / "extract_done.json",
    )
    # NOT [phase=done] — reserved for the dispatcher's single terminal line.
    logger.info("[phase=extract_complete] S3 extraction complete (%.1fs)", time.time() - t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] extraction crashed:\n%s", traceback.format_exc())
        raise
