#!/usr/bin/env python3
"""Build the LMSYS-only quantitative-eval inputs for the #2552 exact replication.

Runs after ``issue2552_exactrep_train.py`` has produced the answer SAE.  The driver
selects 2,000 rows from its 20,000-row holdout, builds answer-SAE and public
per-token max/sum feature lists, and mines eval-disjoint top-25 examples for every
feature description needed by those lists.  Outputs are staged in the schema read
by ``issue2552_exactrep_judge.py``.

The public comparator is andyrdt trainer_2 (layer 19, k=128), the same declared
dictionary-provenance deviation as #2552.  Assistant content spans use the exact
capture tokenizer/offset convention.  Per-token pooling also preserves #2552's
reference token-pool filter (first-eight-position strip + 10x-median norm filter).

Phases are resumable at file granularity:
  select  paired-store/split assertions, deterministic eval and mining pools
  rep     fresh answer-SAE lists + 120k-train-row top-25 mining
  pt      teacher-forced public-SAE lists + 18k non-eval-holdout top-25 mining
  all     select -> rep -> pt

LMSYS text remains under the requested work root and is never printed.  The judge
driver may upload its raw requests/responses under the task's existing policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1482_sae as PUBLIC  # noqa: E402
import issue2476_turnavg_sae as T  # noqa: E402
import issue2552_exactrep_capture as CAP  # noqa: E402
import issue2552_exactrep_prep as PREP  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.atomic_io import write_jsonl_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2552_exactrep_eval_inputs")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)

SEED = 2552
EVAL_N = 2_000
TA_MINE_N = 120_000
MINING_TOP = 25
JUDGED_TOP = 100
TEXT_CAP = 4_000
EXAMPLE_CAP = 1_500
WINDOW_RADIUS = 24
LAYER = 19
CONFIGS = ("rep_ta", "pt_max", "pt_sum")


def _sha_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()


def _paths(args):
    work = Path(args.out_root)
    hf_base = work / "inputs" / args.hf_prefix
    return {
        "work": work,
        "agg": work / "judge_aggregates",
        "lists": hf_base / "analysis_tensors" / "eval_lists",
        "mining": hf_base / "raw_completions" / "mining",
        "selection": work / "selection.npz",
        "selection_meta": work / "selection.json",
        "texts": work / "eval_texts.jsonl",
    }


def _read_splits(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k], np.int64) for k in z.files}


def derive_eval_pools(
    splits: dict[str, np.ndarray], seed: int = SEED, eval_n: int = EVAL_N
) -> dict[str, np.ndarray]:
    """Deterministic disjoint pools from the SAE's own split."""
    holdout = np.asarray(splits["holdout"], np.int64)
    train = np.asarray(splits["train"], np.int64)
    assert len(holdout) >= eval_n
    rng = np.random.default_rng(seed)
    eval_ids = np.sort(rng.choice(holdout, size=eval_n, replace=False))
    pt_mine = np.setdiff1d(holdout, eval_ids, assume_unique=True)
    ta_n = min(TA_MINE_N, len(train))
    ta_mine = np.sort(rng.choice(train, size=ta_n, replace=False))
    assert not np.intersect1d(eval_ids, pt_mine).size
    assert not np.intersect1d(eval_ids, ta_mine).size
    return {"eval": eval_ids, "pt_mine": pt_mine, "ta_mine": ta_mine}


def _row_identities(row_index: Path, positions: np.ndarray) -> dict[tuple[str, int], int]:
    wanted = set(int(x) for x in positions)
    out: dict[tuple[str, int], int] = {}
    with row_index.open(encoding="utf-8") as f:
        for pos, line in enumerate(f):
            if pos not in wanted:
                continue
            rec = json.loads(line)
            assert int(rec["row"]) == pos, (rec["row"], pos)
            key = (str(rec["conversation_id"]), int(rec["msg_idx"]))
            assert key not in out, key
            out[key] = pos
    assert len(out) == len(wanted), (len(out), len(wanted))
    return out


def _scan_texts(corpus_dir: Path, identities: dict[tuple[str, int], int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for path in sorted(corpus_dir.glob("conv_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = str(rec["conversation_id"])
                for mi in rec["asst_msg_idx"]:
                    row = identities.get((cid, int(mi)))
                    if row is not None:
                        out[row] = str(rec["msgs"][mi]["content"])
        if len(out) == len(identities):
            break
    assert len(out) == len(identities), (len(out), len(identities))
    return out


def phase_select(args) -> None:
    p = _paths(args)
    for d in (p["work"], p["agg"], p["lists"], p["mining"]):
        d.mkdir(parents=True, exist_ok=True)
    answer = Path(args.answer_dir)
    context = Path(args.context_dir)
    assert (answer / "row_index.jsonl").read_bytes() == (
        context / "row_index.jsonl"
    ).read_bytes(), "paired row indices differ"
    a_split = _read_splits(answer / "splits.npz")
    c_split = _read_splits(context / "splits.npz")
    assert set(a_split) == set(c_split)
    for k in a_split:
        assert np.array_equal(a_split[k], c_split[k]), f"paired split drift: {k}"
    pools = derive_eval_pools(a_split)
    np.savez(p["selection"], **pools)
    eval_identity = _row_identities(answer / "row_index.jsonl", pools["eval"])
    eval_text = _scan_texts(Path(args.corpus_dir), eval_identity)
    write_jsonl_atomic(
        p["texts"],
        [{"row_id": int(r), "text": eval_text[int(r)][:TEXT_CAP]} for r in pools["eval"]],
    )
    g2 = {
        "descoped": False,
        "n_eval_orig": int(len(pools["eval"])),
        "n_eval_realized": int(len(pools["eval"])),
        "eval_ids": [int(x) for x in pools["eval"]],
        "eval_ids_sha256": _sha_ids(pools["eval"]),
        "orig_eval_ids_sha256": _sha_ids(pools["eval"]),
        "rep_panel_ids": [],
        "rep_panel_sha256": _sha_ids(np.empty(0, np.int64)),
        "rep_panel_n_realized": 0,
        "seed": SEED,
        **as_metadata_dict(git_provenance(), phase="exactrep-eval-select"),
    }
    PREP._write_json_atomic(p["agg"] / "g2_decision.json", g2)
    PREP._write_json_atomic(
        p["selection_meta"],
        {
            "pools": {k: int(len(v)) for k, v in pools.items()},
            "sha256": {k: _sha_ids(v) for k, v in pools.items()},
            "paired_row_index": "PASS",
            "paired_splits": "PASS",
            **as_metadata_dict(git_provenance(), phase="exactrep-eval-select"),
        },
    )
    print(
        f"[select] eval={len(pools['eval'])} ta_mine={len(pools['ta_mine'])} "
        f"pt_mine={len(pools['pt_mine'])}",
        flush=True,
    )


def _load_selection(path: Path) -> dict[str, np.ndarray]:
    assert path.exists(), "run --phase select first"
    with np.load(path) as z:
        return {k: np.asarray(z[k], np.int64) for k in z.files}


def _top_list(vec: torch.Tensor) -> list[list[float | int]]:
    nz = torch.nonzero(vec > 0, as_tuple=False).squeeze(-1)
    if nz.numel() == 0:
        return []
    order = torch.argsort(vec[nz], descending=True)[:JUDGED_TOP]
    ids = nz[order].cpu().tolist()
    vals = vec[nz][order].float().cpu().tolist()
    return [[int(i), float(v)] for i, v in zip(ids, vals, strict=True)]


def _write_list_config(lists_dir: Path, cfg: str, turns: list[dict], meta: dict) -> None:
    lists_dir.mkdir(parents=True, exist_ok=True)
    name = f"lists_{cfg}.jsonl"
    write_jsonl_atomic(lists_dir / name, turns)
    index_path = lists_dir / "feature_lists_2000turns.json"
    doc = json.loads(index_path.read_text()) if index_path.exists() else {"configs": {}}
    doc["configs"][cfg] = {"meta": meta, "files": [name], "n_turns": len(turns)}
    PREP._write_json_atomic(index_path, doc)


@torch.no_grad()
def _mine_ta(sae, mm, positions: np.ndarray, feat_ids: np.ndarray, device: str):
    cols = torch.as_tensor(feat_ids, dtype=torch.long, device=device)
    top_v = torch.full((MINING_TOP, len(feat_ids)), -1.0, device=device)
    top_r = torch.full((MINING_TOP, len(feat_ids)), -1, dtype=torch.long, device=device)
    for chunk_i, s in enumerate(range(0, len(positions), 4096)):
        pos = positions[s : s + 4096]
        x = torch.as_tensor(np.asarray(mm[pos], np.float32), device=device)
        f = sae.encode(x, chunk=2048)[:, cols]
        rows = (
            torch.as_tensor(pos, dtype=torch.long, device=device).unsqueeze(1).expand(-1, len(cols))
        )
        cat_v = torch.cat([top_v, f])
        cat_r = torch.cat([top_r, rows])
        top_v, idx = torch.topk(cat_v, MINING_TOP, dim=0)
        top_r = torch.gather(cat_r, 0, idx)
        if (chunk_i + 1) % 10 == 0:
            print(f"[rep] mining rows={min(s + 4096, len(positions))}/{len(positions)}", flush=True)
    top_r[top_v <= 0] = -1
    return top_v.float().cpu().numpy(), top_r.cpu().numpy()


def phase_rep(args) -> None:
    p = _paths(args)
    pools = _load_selection(p["selection"])
    answer = Path(args.answer_dir)
    mm = np.load(answer / "Y19.fp16.npy", mmap_mode="r")
    sae = T.MatryoshkaBatchTopKSAE.load_local(answer, device=args.device)
    turns: list[dict] = []
    for s in range(0, len(pools["eval"]), 256):
        pos = pools["eval"][s : s + 256]
        x = torch.as_tensor(np.asarray(mm[pos], np.float32), device=args.device)
        f = sae.encode(x)
        turns.extend(
            {"row_id": int(r), "judged_top100": _top_list(f[j])} for j, r in enumerate(pos)
        )
    assert len(turns) == EVAL_N
    _write_list_config(
        p["lists"],
        "rep_ta",
        turns,
        {"list_convention": "all-active answer-turn SAE code, top-100 judged"},
    )
    need = np.asarray(
        sorted({int(fid) for turn in turns for fid, _v in turn["judged_top100"]}), np.int64
    )
    vals, rows = _mine_ta(sae, mm, pools["ta_mine"], need, args.device)
    needed_rows = np.unique(rows[rows >= 0])
    identities = _row_identities(answer / "row_index.jsonl", needed_rows)
    texts = _scan_texts(Path(args.corpus_dir), identities)
    records = []
    for fi, feat in enumerate(need):
        for rank in range(MINING_TOP):
            row = int(rows[rank, fi])
            if row < 0:
                continue
            records.append(
                {
                    "family": "rep_ta",
                    "feat_id": int(feat),
                    "rank": rank,
                    "row_id": row,
                    "activation": float(vals[rank, fi]),
                    "text": texts[row][:EXAMPLE_CAP],
                }
            )
    write_jsonl_atomic(p["mining"] / "top25_rep_ta.jsonl", records)
    print(
        f"[rep] turns={len(turns)} feature_union={len(need)} mining_records={len(records)}",
        flush=True,
    )


def _prepare_selected(rec: dict, tok, selected: dict[tuple[str, int], int]):
    cid = str(rec["conversation_id"])
    wanted = {
        mi: selected[(cid, int(mi))] for mi in rec["asst_msg_idx"] if (cid, int(mi)) in selected
    }
    if not wanted:
        return None
    segs = PREP.render_segments(rec["msgs"], tok)
    n_prefix = len(segs) - len(rec["msgs"])
    full = "".join(segs)
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    roles = [m["role"] for m in rec["msgs"]]
    msg_ids = list(wanted)
    ranges = CAP.content_char_ranges(segs, n_prefix, roles, msg_ids)
    spans = CAP.token_spans_from_offsets(enc["offset_mapping"], ranges)
    selected_spans = []
    for mi, span in zip(msg_ids, spans, strict=True):
        assert span is not None, (cid, mi)
        selected_spans.append((wanted[mi], span))
    return enc["input_ids"], selected_spans


@torch.no_grad()
def _scan_selected_hidden(corpus_dir: Path, selected, tok, model, args, callback) -> None:
    seen: set[int] = set()
    buf = []

    def flush() -> None:
        if not buf:
            return
        lengths = [len(x[0]) for x in buf]
        dev = next(model.parameters()).device
        for batch in CAP.batches_by_budget(lengths, args.batch_max_rows, args.batch_max_tokens):
            max_len = max(lengths[i] for i in batch)
            ids_t = torch.full((len(batch), max_len), tok.pad_token_id, dtype=torch.long)
            mask_t = torch.zeros_like(ids_t)
            for bi, i in enumerate(batch):
                ids = buf[i][0]
                ids_t[bi, : len(ids)] = torch.as_tensor(ids)
                mask_t[bi, : len(ids)] = 1
            captured = CAP.extract_layer_activations(
                model, ids_t.to(dev), [LAYER], attention_mask=mask_t.to(dev)
            )
            hs = captured[LAYER]
            for bi, i in enumerate(batch):
                ids, spans = buf[i]
                full_h = hs[bi, : len(ids)]
                inlier = PUBLIC.token_inlier_mask(full_h)
                inlier[: min(PUBLIC.BOS_OFFSET, len(inlier))] = False
                for row, (s, e) in spans:
                    keep = inlier[s:e]
                    h_span = full_h[s:e]
                    kept = h_span if int(keep.sum()) == 0 else h_span[keep]
                    kept_abs = (
                        torch.arange(s, e, device=keep.device)
                        if int(keep.sum()) == 0
                        else torch.nonzero(keep, as_tuple=False).squeeze(-1) + s
                    )
                    callback(int(row), kept, kept_abs, ids, (s, e))
                    seen.add(int(row))
            del captured, hs
        buf.clear()

    for path in sorted(corpus_dir.glob("conv_*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                prep = _prepare_selected(json.loads(line), tok, selected)
                if prep is not None:
                    buf.append(prep)
                if len(buf) >= 256:
                    flush()
        if len(seen) == len(selected):
            break
    flush()
    assert seen == set(selected.values()), (len(seen), len(selected))


def phase_pt(args) -> None:
    p = _paths(args)
    pools = _load_selection(p["selection"])
    row_index = Path(args.answer_dir) / "row_index.jsonl"
    eval_identity = _row_identities(row_index, pools["eval"])
    mine_identity = _row_identities(row_index, pools["pt_mine"])
    tok, model = CAP.load_capture_model(argparse.Namespace(tiny_model=False, device=args.device))
    pt = PUBLIC.BatchTopKSAE.load(k=128, device=args.device, layer=LAYER)
    eval_rows: dict[str, list[dict]] = {"pt_max": [], "pt_sum": []}

    def eval_callback(row, kept, _kept_abs, _ids, _span):
        f = pt.encode(kept.to(args.device))
        pooled = {"pt_max": f.max(0).values, "pt_sum": f.sum(0)}
        for cfg, vec in pooled.items():
            eval_rows[cfg].append({"row_id": row, "judged_top100": _top_list(vec)})

    _scan_selected_hidden(Path(args.corpus_dir), eval_identity, tok, model, args, eval_callback)
    for cfg in ("pt_max", "pt_sum"):
        eval_rows[cfg].sort(key=lambda r: r["row_id"])
        assert len(eval_rows[cfg]) == EVAL_N
        _write_list_config(
            p["lists"],
            cfg,
            eval_rows[cfg],
            {
                "list_convention": f"public trainer_2 per-token {cfg.removeprefix('pt_')} pool, top-100 judged"
            },
        )
    need = np.asarray(
        sorted(
            {
                int(fid)
                for cfg in ("pt_max", "pt_sum")
                for turn in eval_rows[cfg]
                for fid, _v in turn["judged_top100"]
            }
        ),
        np.int64,
    )
    need_t = torch.as_tensor(need, dtype=torch.long, device=args.device)
    top_v = torch.full((MINING_TOP, len(need)), -1.0, device=args.device)
    top_r = torch.full((MINING_TOP, len(need)), -1, dtype=torch.long, device=args.device)
    top_p = torch.full_like(top_r, -1)

    def mine_callback(row, kept, kept_abs, _ids, _span):
        nonlocal top_v, top_r, top_p
        f = pt.encode(kept.to(args.device))[:, need_t]
        row_v, peak = f.max(0)
        row_p = kept_abs.to(args.device)[peak]
        cat_v = torch.cat([top_v, row_v.unsqueeze(0)])
        cat_r = torch.cat(
            [top_r, torch.full((1, len(need)), row, dtype=torch.long, device=args.device)]
        )
        cat_p = torch.cat([top_p, row_p.unsqueeze(0)])
        top_v, idx = torch.topk(cat_v, MINING_TOP, dim=0)
        top_r = torch.gather(cat_r, 0, idx)
        top_p = torch.gather(cat_p, 0, idx)

    _scan_selected_hidden(Path(args.corpus_dir), mine_identity, tok, model, args, mine_callback)
    top_r[top_v <= 0] = -1
    wanted_by_row: dict[int, list[tuple[int, int, float, int]]] = {}
    v_np, r_np, p_np = top_v.cpu().numpy(), top_r.cpu().numpy(), top_p.cpu().numpy()
    for fi, feat in enumerate(need):
        for rank in range(MINING_TOP):
            row = int(r_np[rank, fi])
            if row >= 0:
                wanted_by_row.setdefault(row, []).append(
                    (int(feat), rank, float(v_np[rank, fi]), int(p_np[rank, fi]))
                )
    emit_identity = _row_identities(row_index, np.asarray(sorted(wanted_by_row), np.int64))
    records: list[dict] = []

    def emit_callback(row, kept, kept_abs, ids, span):
        wants = wanted_by_row[row]
        feat_ids = torch.as_tensor([w[0] for w in wants], dtype=torch.long, device=args.device)
        x = kept.to(device=args.device, dtype=torch.float32)
        acts = torch.relu((x - pt.b_dec) @ pt.w_enc[feat_ids].T + pt.b_enc[feat_ids])
        acts *= acts > pt.threshold
        abs_to_kept = {int(a): i for i, a in enumerate(kept_abs.cpu().tolist())}
        for wi, (feat, rank, value, peak_abs) in enumerate(wants):
            lo = max(span[0], peak_abs - WINDOW_RADIUS)
            hi = min(span[1], peak_abs + WINDOW_RADIUS + 1)
            pairs = []
            for abs_pos in range(lo, hi):
                ki = abs_to_kept.get(abs_pos)
                if ki is not None and float(acts[ki, wi]) > 0:
                    pairs.append([abs_pos - lo, float(acts[ki, wi])])
            records.append(
                {
                    "family": "pt",
                    "feat_id": feat,
                    "rank": rank,
                    "row_id": row,
                    "activation": value,
                    "peak_token_abs": peak_abs,
                    "window_lo_abs": lo,
                    "window_token_acts": pairs,
                    "window_text": tok.decode(ids[lo:hi], skip_special_tokens=False),
                }
            )

    _scan_selected_hidden(Path(args.corpus_dir), emit_identity, tok, model, args, emit_callback)
    records.sort(key=lambda r: (r["feat_id"], r["rank"]))
    write_jsonl_atomic(p["mining"] / "top25_pt.jsonl", records)
    print(
        f"[pt] turns={EVAL_N} feature_union={len(need)} mining_records={len(records)}", flush=True
    )


PHASES = {"select": phase_select, "rep": phase_rep, "pt": phase_pt}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=["all", *PHASES], default="all")
    ap.add_argument("--corpus-dir", type=Path, default=Path("/workspace/eps-2552-exactrep/corpus"))
    ap.add_argument("--answer-dir", type=Path, default=Path("/workspace/eps-2552-exactrep/sae_rep"))
    ap.add_argument(
        "--context-dir", type=Path, default=Path("/workspace/eps-2552-exactrep/sae_ctx_rep")
    )
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/eps-2552-exactrep/judge"))
    ap.add_argument("--hf-prefix", default="issue2552_derreplication/exactrep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-max-rows", type=int, default=8)
    ap.add_argument("--batch-max-tokens", type=int, default=16_384)
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        return 0
    if args.list_phases:
        print(sorted(PHASES))
        return 0
    names = list(PHASES) if args.phase == "all" else [args.phase]
    for name in names:
        PHASES[name](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
