"""#1776 Phase 3: steered-regeneration ground truth (plan v4 §4 Phase 3 / §5 / §6).

Grid: contexts × α grid × directions × K samples (+ the α=0 baseline cell).
Mechanism: the reused ``DeltaHook(layer=SOURCE_LAYER)`` PREFILL-ONLY mode — it
edits ONLY position T−1 (each row's LAST real context token under LEFT padding)
of the block-14 output, exactly the tensor slot ``J_last`` differentiates and
M′ reads (§4 slot-pinning block). Generation via the reused batched
``generate_batch`` (one HF ``generate`` per draw, batched over contexts).

Persist order per stratum (direction, α) — checkpoint-per-phase, #779 rule:
  1. generate → WRITE rollout text JSON (``raw_completions/steered/``) BEFORE
     any reduce (the pod dispatcher uploads these to
     ``issue1776_jacobian/raw_completions/steered/`` before capture/judging);
  2. teacher-forced capture of each generation's v_{L′=19} (UNHOOKED — the
     same rig as Phase 0.4/#779; the measured Δv̄ is the text-mediated shift,
     the plan's teacher-forced-vs-regeneration measurement note);
  3. per-sample summaries persisted (``summaries/<stratum>.pt``);
  4. per-cell table rows (``cells/<stratum>.jsonl``) carrying context ids so
     the §6 context-level clustered bootstrap is a pure re-reduction.

Prediction legs are FIXED per (direction, α) stratum (§6 statistical block):
``pred_J = J_last @ (αΔ)`` and ``pred_M′ = ((αΔ)/xsd) @ W`` (the standardized
ridge's linear response — xmu/ymu cancel in the differential). Per-cell reads:
cos(Δv̄, pred) + magnitude ratio, Δv̄ = mean v19(steered) − mean v19(α=0).

Coherence (#1415 ≥50% rule, gate G-COHERENCE): per-cell coherent flags +
``coherence_pass`` are RECORDED; incoherent cells are excluded downstream via
the column (never dropped from the table) and counted. The gate never kills
the phase. A per-direction operating-α report (largest passing α — walking
down the geometric grid IS the halve-once rule) is emitted in the summary.

Resume: a manifest pins EVERY output-affecting regime key; a mismatch REFUSES
(fresh --out-root required — silent recompute could mix regimes inside the
raw-completion files). Done strata are skipped via state/<stratum>.done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import issue1776_common as C76  # noqa: F401  (sys.path side-effect + helpers)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)
import numpy as np
import torch

import issue779_common as C

from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    DeltaHook,
    capture_vectors,
    coherence_check,
    condition_passes,
    generate_batch,
)

TRAIT_DIRECTIONS = ("evil", "sycophancy", "hallucination")
DEFAULT_DIRECTIONS = (*TRAIT_DIRECTIONS, "w1_mprime", "random")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.to(torch.float32).contiguous().numpy().tobytes()).hexdigest()


def stratum_key(direction: str, alpha: float, mode: str = "prefill") -> str:
    base = f"{direction}_a{alpha:g}"
    return base if mode == "prefill" else f"{base}_allpos"


# ── inputs ────────────────────────────────────────────────────────────────────


def load_contexts(path: Path, limit: int = 0) -> list[dict]:
    """Context rows: {"context_id","user"[,"system","source"]} — pair-manifest
    rows {"pair_id","prompt",...} (the round-B interface) are accepted and
    mapped (context_id=pair_id, user=prompt; the stored response is ignored)."""
    rows: list[dict] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "context_id" not in r and "pair_id" in r:
                r = {"context_id": str(r["pair_id"]), "user": r["prompt"], **r}
            assert "context_id" in r and ("user" in r or "prompt" in r), sorted(r)
            cid = str(r["context_id"])
            assert cid not in seen, f"duplicate context_id {cid!r}"
            seen.add(cid)
            rows.append(
                {
                    "context_id": cid,
                    "user": r.get("user") or r["prompt"],
                    "system": r.get("system") or None,
                    "source": r.get("source", "unspecified"),
                }
            )
    assert rows, f"no contexts in {path}"
    if limit:
        rows = rows[:limit]
    return rows


def load_directions(args, hidden: int) -> tuple[dict[str, torch.Tensor], dict]:
    """Direction bank Δ (unit-normed, layer SOURCE_LAYER input space), plan §4.

    - r_B traits: the LAYER-``source_layer`` row of the (L, H) per-trait stack
      (#779 ``r_b/{trait}.pt``; row via the stored ``layers`` list).
    - ``w1_mprime``: top input-space singular direction of M′'s RAW-space
      operator A = W/xsd[:,None] (pred_shift = Δ @ A), i.e. U[:,0] of svd(A).
    - ``random``: seeded unit-norm Gaussian (norm-matched control — all
      directions are unit-normed).
    """
    bank: dict[str, torch.Tensor] = {}
    prov: dict[str, dict] = {}
    for name in args.directions:
        if name in TRAIT_DIRECTIONS:
            d = torch.load(args.rb_dir / f"{name}.pt", map_location="cpu", weights_only=True)
            layers = [int(x) for x in d["layers"]]
            assert args.source_layer in layers, (args.source_layer, layers)
            vec = d["r_b"][layers.index(args.source_layer)].to(torch.float32)
            prov[name] = {"file": str(args.rb_dir / f"{name}.pt"), "layer": args.source_layer}
        elif name == "w1_mprime":
            payload = torch.load(args.mprime_weights, map_location="cpu", weights_only=True)
            w = payload["W"].to(torch.float64)
            xsd = payload["xsd"].to(torch.float64)
            assert w.shape[0] == xsd.shape[0] == hidden, (w.shape, xsd.shape, hidden)
            a = w / xsd[:, None]  # raw-space operator, (d_in, d_out)
            u, s, _ = torch.linalg.svd(a, full_matrices=False)
            vec = u[:, 0].to(torch.float32)
            prov[name] = {"file": str(args.mprime_weights), "top_sv": float(s[0])}
        elif name == "random":
            g = torch.Generator().manual_seed(args.random_seed)
            vec = torch.randn(hidden, generator=g)
            prov[name] = {"seed": args.random_seed}
        else:
            raise ValueError(f"unknown direction {name!r}")
        assert vec.shape == (hidden,), (name, vec.shape, hidden)
        n = float(vec.norm())
        assert n > 0, f"direction {name} has zero norm"
        bank[name] = (vec / n).to(torch.float32)
    return bank, prov


def prediction_legs(
    args, bank: dict[str, torch.Tensor], hidden: int
) -> tuple[dict[str, torch.Tensor], dict]:
    """Fixed per-(direction, α) prediction vectors for BOTH operators (§6)."""
    jd = torch.load(args.jlast, map_location="cpu", weights_only=True)
    j = jd["J"].to(torch.float64)
    assert j.shape == (hidden, hidden), (
        f"J_last must be the merged FULL-RANK (H, H) matrix, got {tuple(j.shape)} "
        "(sketch-mode J is not a phase-3 predictor — plan §4 full-rank directive)"
    )
    payload = torch.load(args.mprime_weights, map_location="cpu", weights_only=True)
    w = payload["W"].to(torch.float64)
    xsd = payload["xsd"].to(torch.float64)
    assert w.shape[0] == hidden and xsd.shape == (hidden,), (w.shape, xsd.shape)
    preds: dict[str, torch.Tensor] = {}
    for name, vec in bank.items():
        v = vec.to(torch.float64)
        for a in args.alphas:
            key = f"{name}_a{a:g}"
            pj = j @ (a * v)
            pm = ((a * v) / xsd) @ w
            preds[key] = torch.stack([pj, pm]).to(torch.float32)  # (2, H): [J_last, M']
    meta = {
        "jlast_sha": _sha256_tensor(j.to(torch.float32)),
        "mprime_w_sha": _sha256_tensor(w.to(torch.float32)),
        "mprime_selected_lambda": float(payload.get("selected_lambda", math.nan)),
        "legs": ["J_last", "M'"],
    }
    return preds, meta


# ── model ─────────────────────────────────────────────────────────────────────


def load_model(args):
    if args.tiny:
        import issue1776_jlens_fit as JF

        _, model, tok = JF.load_lens_model(C.DEFAULT_MODEL, device="cpu", tiny=True)
        return model, tok
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), device_map=args.device
    ).eval()
    return model, tok


# ── manifest / resume ────────────────────────────────────────────────────────


def build_manifest(args, contexts: list[dict], bank: dict[str, torch.Tensor]) -> dict:
    """EVERY output-affecting regime key (RNG consumption depends on chunking +
    context order; capture numerics on batch geometry — resume-pin lesson)."""
    order_sha = hashlib.sha256("\n".join(c["context_id"] for c in contexts).encode()).hexdigest()
    dir_sha = hashlib.sha256(b"".join(bank[n].numpy().tobytes() for n in sorted(bank))).hexdigest()
    return {
        "script": "issue1776_phase3",
        "model": args.model,
        "tiny": bool(args.tiny),
        "source_layer": args.source_layer,
        "readout_layer": args.readout_layer,
        "alphas": [float(a) for a in args.alphas],
        "k_samples": args.k_samples,
        "k_baseline": args.k_baseline,
        "temperature": args.temperature,
        "seed_base": args.seed_base,
        "max_new_tokens": args.max_new_tokens,
        "gen_batch": args.gen_batch,
        "capture_batch": args.capture_batch,
        "contexts_sha": _sha256_file(args.contexts),
        "context_order_sha": order_sha,
        "n_contexts": len(contexts),
        "directions": sorted(bank),
        "directions_sha": dir_sha,
        "random_seed": args.random_seed,
        "all_positions_subset": args.all_positions_subset,
        "all_positions_alphas": [float(a) for a in args.all_positions_alphas],
    }


def check_manifest(out_root: Path, manifest: dict) -> None:
    path = out_root / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text())
        prior.pop("repro", None)
        if prior != manifest:
            diff = {k for k in {*prior, *manifest} if prior.get(k) != manifest.get(k)}
            raise RuntimeError(
                f"manifest MISMATCH on resume (keys: {sorted(diff)}) — regimes must not mix "
                f"inside one out-root; use a fresh --out-root. prior={path}"
            )
    C76.atomic_write_json(path, manifest)


# ── per-stratum execution ────────────────────────────────────────────────────


def _nonempty_idx(tok, texts: list[str]) -> list[int]:
    """Sample indices kept by the capture rig's own criterion (>=1 token)."""
    return [
        i for i, t in enumerate(texts) if len(tok(t, add_special_tokens=False)["input_ids"]) > 0
    ]


def generate_stratum(
    model, tok, contexts: list[dict], args, *, delta: torch.Tensor | None, alpha: float, mode: str
) -> tuple[list[list[str]], int]:
    """Batched generation over context chunks; returns texts[ctx][sample] + n hook edits."""
    k = args.k_baseline if delta is None else args.k_samples
    texts: list[list[str]] = []
    n_edits = 0
    for start in range(0, len(contexts), args.gen_batch):
        chunk = contexts[start : start + args.gen_batch]
        ctx_dicts = [{"system": c["system"], "user": c["user"]} for c in chunk]
        if delta is None:
            res = generate_batch(
                model,
                tok,
                ctx_dicts,
                n=k,
                hook=None,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed_base=args.seed_base,
            )
        else:
            with DeltaHook(
                model,
                args.source_layer,
                delta,
                alpha,
                all_positions=(mode == "all_positions"),
            ) as hook:
                res = generate_batch(
                    model,
                    tok,
                    ctx_dicts,
                    n=k,
                    hook=hook,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    seed_base=args.seed_base,
                )
                n_edits += hook.n_edits
        texts.extend(res)
        print(
            f"[phase3] gen stratum done chunk {start // args.gen_batch + 1}/"
            f"{(len(contexts) + args.gen_batch - 1) // args.gen_batch} "
            f"(alpha={alpha:g} mode={mode})",
            flush=True,
        )
    assert len(texts) == len(contexts)
    return texts, n_edits


def capture_stratum(
    model, tok, contexts: list[dict], texts: list[list[str]], args
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    """Teacher-forced UNHOOKED v_{L'} per non-empty sample; ragged per context."""
    kept_idx = {c["context_id"]: _nonempty_idx(tok, t) for c, t in zip(contexts, texts)}
    keep = [i for i, c in enumerate(contexts) if kept_idx[c["context_id"]]]
    v19: dict[str, torch.Tensor] = {}
    for start in range(0, len(keep), args.gen_batch):
        sel = keep[start : start + args.gen_batch]
        ctx_dicts = [{"system": contexts[i]["system"], "user": contexts[i]["user"]} for i in sel]
        comps = [[texts[i][j] for j in kept_idx[contexts[i]["context_id"]]] for i in sel]
        cap = capture_vectors(
            model,
            tok,
            ctx_dicts,
            [args.readout_layer],
            completions=comps,
            batch_size=args.capture_batch,
        )
        for i, rec in zip(sel, cap["per_context"]):
            cid = contexts[i]["context_id"]
            per = rec["v_a_per_completion"][:, 0, :]  # (n_kept, H) at the single layer
            assert per.shape[0] == len(kept_idx[cid]), (per.shape, len(kept_idx[cid]))
            v19[cid] = per.to(torch.float32)
    return v19, kept_idx


def cell_rows(
    contexts: list[dict],
    texts: list[list[str]],
    v19: dict[str, torch.Tensor],
    kept_idx: dict[str, list[int]],
    base_means: dict[str, torch.Tensor] | None,
    pred: torch.Tensor | None,
    *,
    direction: str,
    alpha: float,
    mode: str,
) -> list[dict]:
    """One row per (context, direction, α) cell — the §6 bootstrap unit."""
    rows = []
    for c, samp in zip(contexts, texts):
        cid = c["context_id"]
        kept = kept_idx.get(cid, [])
        flags = coherence_check(samp)
        row: dict = {
            "cell_id": f"{stratum_key(direction, alpha, mode)}__{cid}",
            "mode": mode,
            "direction": direction,
            "alpha": float(alpha),
            "context_id": cid,
            "source": c["source"],
            "n_samples": len(samp),
            "n_empty": len(samp) - len(kept),
            "n_coherent": int(sum(flags)),
            "coherence_pass": bool(condition_passes(flags)),
            "n_kept_capture": len(kept),
        }
        if base_means is not None:  # steered stratum
            base = base_means.get(cid)
            row["baseline_missing"] = base is None
            if base is not None and kept:
                dv = (v19[cid].to(torch.float64).mean(dim=0) - base.to(torch.float64)).numpy()
                dvn = float(np.linalg.norm(dv))
                row["dv_norm"] = dvn
                for leg, name in ((0, "jlast"), (1, "mprime")):
                    p = pred[leg].to(torch.float64).numpy()
                    pn = float(np.linalg.norm(p))
                    ok = dvn > 0 and pn > 0
                    row[f"cos_pred_{name}"] = float(np.dot(dv, p) / (dvn * pn)) if ok else None
                    row[f"mag_ratio_{name}"] = float(dvn / pn) if pn > 0 else None
            else:
                row["dv_norm"] = None
                for name in ("jlast", "mprime"):
                    row[f"cos_pred_{name}"] = None
                    row[f"mag_ratio_{name}"] = None
        rows.append(row)
    return rows


def run_stratum(
    model,
    tok,
    contexts: list[dict],
    args,
    dirs: dict[str, Path],
    *,
    direction: str,
    alpha: float,
    mode: str,
    delta: torch.Tensor | None,
    pred: torch.Tensor | None,
    base_means: dict[str, torch.Tensor] | None,
) -> None:
    key = stratum_key(direction, alpha, mode)
    done = dirs["state"] / f"{key}.done"
    if done.exists():
        print(f"[phase3] stratum {key} already done — skip (resume)", flush=True)
        return
    t0 = time.time()
    texts, n_edits = generate_stratum(
        model, tok, contexts, args, delta=delta, alpha=alpha, mode=mode
    )
    if delta is not None:
        assert n_edits > 0, f"stratum {key}: DeltaHook never fired"
    # 1) rollout text persists BEFORE any reduce (#779 rule).
    raw = {
        "stratum": key,
        "direction": direction,
        "alpha": float(alpha),
        "mode": mode,
        "model": args.model,
        "n_hook_edits": n_edits,
        "contexts": [{**c, "samples": samp} for c, samp in zip(contexts, texts, strict=True)],
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(dirs["raw"] / f"{key}.json", raw)
    # 2) teacher-forced v_{L'} capture + 3) per-sample summaries.
    v19, kept_idx = capture_stratum(model, tok, contexts, texts, args)
    torch.save(
        {
            "stratum": key,
            "layer": args.readout_layer,
            "context_ids": [c["context_id"] for c in contexts],
            "v19": v19,
            "kept_sample_idx": kept_idx,
        },
        dirs["summaries"] / f"{key}.pt",
    )
    # 4) per-cell table (whole-file write per stratum — clean resume unit).
    rows = cell_rows(
        contexts,
        texts,
        v19,
        kept_idx,
        base_means,
        pred,
        direction=direction,
        alpha=alpha,
        mode=mode,
    )
    with open(dirs["cells"] / f"{key}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    done.write_text(json.dumps({"elapsed_s": time.time() - t0, "n_rows": len(rows)}))
    print(
        f"[phase3] [phase=stratum_done key={key}] rows={len(rows)} "
        f"hook_edits={n_edits} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )


def load_base_means(dirs: dict[str, Path]) -> dict[str, torch.Tensor]:
    st = torch.load(dirs["summaries"] / "baseline_a0.pt", map_location="cpu", weights_only=True)
    return {cid: v.to(torch.float64).mean(dim=0) for cid, v in st["v19"].items()}


# ── driver ────────────────────────────────────────────────────────────────────


def run(args) -> int:
    # Cheap-input validation BEFORE the heavy 7B load (review v1 Minor: a bad
    # --contexts path must fail in seconds, not after model init).
    contexts = load_contexts(args.contexts, args.limit_contexts)
    model, tok = load_model(args)
    hidden = (model.model if hasattr(model, "model") else model).config.hidden_size
    bank, dir_prov = load_directions(args, hidden)
    preds, pred_meta = prediction_legs(args, bank, hidden)

    out = args.out_root
    dirs = {
        "raw": out / "raw_completions" / "steered",
        "summaries": out / "summaries",
        "cells": out / "cells",
        "state": out / "state",
        "eval": args.eval_out,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args, contexts, bank)
    check_manifest(out, manifest)
    # Existence-gated: under --strata-num-shards>1 the concurrent shards would
    # otherwise race identical NON-atomic torch.save writes (the dispatcher's
    # single-process --baseline-only pass writes these first).
    if not (out / "directions.pt").exists():
        torch.save({"bank": bank, "provenance": dir_prov}, out / "directions.pt")
    if not (out / "predictions.pt").exists():
        torch.save({"preds": preds, "meta": pred_meta}, out / "predictions.pt")

    # Baseline FIRST (every steered cell's Δv̄ + the judge contrast need it).
    # Under sharding the dispatcher runs a --baseline-only pass BEFORE the
    # fan-out; the .done resume file then skips it inside every shard.
    run_stratum(
        model,
        tok,
        contexts,
        args,
        dirs,
        direction="baseline",
        alpha=0.0,
        mode="prefill",
        delta=None,
        pred=None,
        base_means=None,
    )
    if args.baseline_only:
        print("[phase3] baseline-only pass complete", flush=True)
        return 0
    base_means = load_base_means(dirs)

    strata: list[tuple[str, float, str]] = [
        (name, float(a), "prefill") for name in bank for a in args.alphas
    ]
    if args.all_positions_subset > 0:
        strata += [
            (name, float(a), "all_positions") for name in bank for a in args.all_positions_alphas
        ]
    if args.strata_num_shards > 1:
        assert 0 <= args.strata_shard < args.strata_num_shards, (
            args.strata_shard,
            args.strata_num_shards,
        )
        strata = [
            s for i, s in enumerate(strata) if i % args.strata_num_shards == args.strata_shard
        ]
        print(
            f"[phase3] shard {args.strata_shard}/{args.strata_num_shards}: {len(strata)} strata "
            f"{[stratum_key(n, a, m) for n, a, m in strata]}",
            flush=True,
        )
    for name, a, mode in strata:
        ctxs = contexts if mode == "prefill" else contexts[: args.all_positions_subset]
        run_stratum(
            model,
            tok,
            ctxs,
            args,
            dirs,
            direction=name,
            alpha=a,
            mode=mode,
            delta=bank[name],
            pred=preds[f"{name}_a{a:g}"],
            base_means=base_means,
        )

    if args.strata_num_shards > 1:
        # Partial cells under this shard — the dispatcher runs --finalize-only
        # once every shard has exited 0.
        print(f"[phase3] shard {args.strata_shard} complete (finalize deferred)", flush=True)
        return 0
    return finalize(args, dirs, bank, manifest)


def finalize_only(args) -> int:
    """Aggregate an already-sharded out_root (no model load): bank names +
    manifest come from the run's own persisted directions.pt / manifest.json."""
    out = args.out_root
    bank = torch.load(out / "directions.pt", map_location="cpu", weights_only=False)["bank"]
    manifest = json.loads((out / "manifest.json").read_text())
    n_strata = len(bank) * len(args.alphas)
    done = list((out / "state").glob("*.done"))
    n_expected = 1 + n_strata  # baseline + prefill strata (all_positions extra tolerated)
    assert len(done) >= n_expected, (
        f"finalize-only: {len(done)} .done strata < expected {n_expected} — shards incomplete"
    )
    dirs = {
        "raw": out / "raw_completions" / "steered",
        "summaries": out / "summaries",
        "cells": out / "cells",
        "state": out / "state",
        "eval": args.eval_out,
    }
    dirs["eval"].mkdir(parents=True, exist_ok=True)
    return finalize(args, dirs, bank, manifest)


def _relocate_misnested_eval_dir(eval_dir: Path, out_root: Path) -> None:
    """Crash-fix r9: a pre-fix dispatch passed the deliverable FILE path as
    --eval-out, so finalize wrote eval_dir/steered_shift_summaries.json/ as a
    DIRECTORY with the real outputs nested one level deep. Relocate EXACTLY
    that shape into out_root scratch so the corrected FILE write can land
    (atomic_write_json's os.replace fails on an existing dir). Any OTHER
    shape at the path fails LOUD — never an unconditional delete under
    eval_results/. Idempotent: no-op once the dir is gone."""
    bad = eval_dir / "steered_shift_summaries.json"
    if not bad.is_dir():
        return
    allowed = {"steered_shift_summaries.json", "raw_completions_manifest.json"}
    inner = {p.name for p in bad.iterdir()}
    if not inner <= allowed or any(not (bad / n).is_file() for n in inner):
        raise RuntimeError(
            f"misnest-repair: unexpected contents at {bad}: {sorted(inner)} "
            f"(expected regular-file subset of {sorted(allowed)}) — refusing to touch it"
        )
    dest = out_root / "misnested_eval_out" / f"steered_shift_summaries.json.{int(time.time())}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(bad), str(dest))
    print(f"[phase3] [misnest-repair] relocated pre-fix eval-out DIR {bad} -> {dest}", flush=True)


def finalize(args, dirs: dict[str, Path], bank: dict[str, torch.Tensor], manifest: dict) -> int:
    """Aggregate the per-cell tables → steered_shift_summaries.json + raw manifest."""
    _relocate_misnested_eval_dir(dirs["eval"], args.out_root)
    all_rows: list[dict] = []
    for p in sorted(dirs["cells"].glob("*.jsonl")):
        with open(p) as f:
            all_rows.extend(json.loads(ln) for ln in f if ln.strip())
    per_stratum: dict[str, dict] = {}
    op_alpha: dict[str, float | None] = {}
    for name in bank:
        passing = {}
        for a in args.alphas:
            rows = [
                r
                for r in all_rows
                if r["direction"] == name and r["alpha"] == float(a) and r["mode"] == "prefill"
            ]
            key = stratum_key(name, float(a))
            coh = [r for r in rows if r["coherence_pass"]]
            ok = [r for r in coh if r.get("cos_pred_jlast") is not None]
            per_stratum[key] = {
                "n_cells": len(rows),
                "n_coherent_cells": len(coh),
                "n_scored_cells": len(ok),
                "mean_cos_jlast": float(np.mean([r["cos_pred_jlast"] for r in ok])) if ok else None,
                "mean_cos_mprime": float(np.mean([r["cos_pred_mprime"] for r in ok]))
                if ok
                else None,
                "mean_dv_norm": float(np.mean([r["dv_norm"] for r in ok])) if ok else None,
            }
            passing[a] = len(coh) / len(rows) >= 0.5 if rows else False
        # walking DOWN the geometric grid IS the #1415 halve-once rule
        op_alpha[name] = next((a for a in sorted(args.alphas, reverse=True) if passing[a]), None)
    summary = {
        "manifest": manifest,
        "cells_table_dir": str(dirs["cells"]),
        "summaries_dir": str(dirs["summaries"]),
        "per_stratum": per_stratum,
        "operating_alpha": op_alpha,
        "n_cell_rows_total": len(all_rows),
        "bootstrap_note": (
            "context-level clustered bootstrap (plan §6) re-reduces cells/*.jsonl: resample "
            "context_ids with replacement carrying ALL their cells; direction is a FIXED "
            "factor; steer_rand ('random') is reported separately, never pooled"
        ),
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(dirs["eval"] / "steered_shift_summaries.json", summary)
    raw_files = sorted(
        p.name for p in (args.out_root / "raw_completions" / "steered").glob("*.json")
    )
    C76.atomic_write_json(
        dirs["eval"] / "raw_completions_manifest.json",
        {
            "local_dir": str(args.out_root / "raw_completions" / "steered"),
            "hf_dest_prefix": f"{C76.HF_PREFIX}/raw_completions/steered/",
            "files": raw_files,
            "note": "uploaded by the pod dispatcher BEFORE capture/judging (plan §9 order)",
            "repro": C76.repro_meta(),
        },
    )
    # Deliverables must be regular FILES at the plan-§6.5 paths — .exists()
    # alone is satisfied by the r8 misnested DIRECTORY shape (crash-fix r9).
    for name in ("steered_shift_summaries.json", "raw_completions_manifest.json"):
        dp = dirs["eval"] / name
        assert dp.is_file(), f"phase3 deliverable missing or not a regular FILE: {dp}"
    print(
        f"[phase3] [phase=phase3_done] strata={len(raw_files)} cell_rows={len(all_rows)} "
        f"-> {dirs['eval']}",
        flush=True,
    )
    return 0


# ── tiny-real CPU smoke ───────────────────────────────────────────────────────


def smoke(args) -> int:
    """Full cell-loop e2e on the from-config tiny Qwen2 (2 ctx × 1 α × 2 dir × K=2)."""
    out = args.out_root
    out.mkdir(parents=True, exist_ok=True)
    args.tiny = True
    args.source_layer, args.readout_layer = 1, 3
    # α=50 + temperature=0.05 (NOT production values): the random tiny net's
    # 0.02-init lm_head yields near-uniform 150k-way logits, so under the
    # shared per-draw RNG a temp-1.0 multinomial flips NO token even at
    # α=5000 (measured) — identical texts ⇒ dv=0 ⇒ the None-cos guard fires
    # instead of the cos path. A low-but-nonzero temperature keeps the REAL
    # do_sample=True branch (production temp 1.0) while concentrating the
    # distribution so the direction change actually flips tokens (4/4).
    args.alphas, args.k_samples, args.k_baseline = [50.0], 2, 2
    args.temperature = 0.05
    args.directions = ["evil", "w1_mprime"]
    args.gen_batch, args.capture_batch, args.max_new_tokens = 2, 2, 8
    args.limit_contexts, args.all_positions_subset = 0, 0

    hidden = 64
    args.contexts = out / "smoke_contexts.jsonl"
    args.contexts.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"context_id": "c0", "user": "What is the capital of France?", "source": "smoke"},
                {"pair_id": "c1", "prompt": "Name one prime number.", "source": "smoke"},
            )
        )
        + "\n"
    )
    g = torch.Generator().manual_seed(7)
    args.rb_dir = out
    torch.save(
        {"trait": "evil", "r_b": torch.randn(4, hidden, generator=g), "layers": [0, 1, 2, 3]},
        out / "evil.pt",
    )
    args.mprime_weights = out / "m_smoke.pt"
    torch.save(
        {
            "kind": "ridge",
            "selected_lambda": 0.1,
            "W": torch.randn(hidden, hidden, generator=g),
            "xsd": torch.rand(hidden, generator=g) + 0.5,
            "xmu": torch.zeros(hidden),
            "ymu": torch.zeros(hidden),
        },
        args.mprime_weights,
    )
    args.jlast = out / "j_smoke.pt"
    torch.save({"J": torch.randn(hidden, hidden, generator=g)}, args.jlast)
    args.eval_out = out / "eval"

    rc = run(args)
    assert rc == 0, rc
    # exact table shape: 2 ctx × (2 dir × 1 α + baseline) = 6 rows
    rows = []
    for p in sorted((out / "cells").glob("*.jsonl")):
        rows.extend(json.loads(ln) for ln in open(p) if ln.strip())
    assert len(rows) == 6, len(rows)
    steered = [r for r in rows if r["direction"] != "baseline"]
    assert len(steered) == 4 and all(not r["baseline_missing"] for r in steered)
    assert all(r["cos_pred_jlast"] is not None for r in steered), steered
    for key in ("baseline_a0", "evil_a50", "w1_mprime_a50"):
        raw = json.loads((out / "raw_completions" / "steered" / f"{key}.json").read_text())
        assert len(raw["contexts"]) == 2 and all(len(c["samples"]) == 2 for c in raw["contexts"])
        if key != "baseline_a0":
            assert raw["n_hook_edits"] >= 2, (key, raw["n_hook_edits"])  # 1 prefill edit/draw
        st = torch.load(out / "summaries" / f"{key}.pt", map_location="cpu", weights_only=True)
        assert all(v.shape[1] == hidden for v in st["v19"].values())
    # is_file, not exists: a misnested DIRECTORY must never satisfy this (r9)
    assert (args.eval_out / "steered_shift_summaries.json").is_file()
    assert (args.eval_out / "raw_completions_manifest.json").is_file()
    print("[phase3] [smoke] e2e cell loop PASS (6 rows, hook fired, persists complete)")

    # resume MATCH: re-run skips every stratum (state markers persist).
    n_done_before = len(list((out / "state").glob("*.done")))
    rc = run(args)
    assert rc == 0 and len(list((out / "state").glob("*.done"))) == n_done_before
    print("[phase3] [smoke] resume MATCH branch: all strata skipped")

    # resume MISMATCH: perturbed regime key REFUSES (fail-loud, no regime mixing).
    man = json.loads((out / "manifest.json").read_text())
    man["seed_base"] = 999
    C76.atomic_write_json(out / "manifest.json", man)
    try:
        run(args)
        raise AssertionError("manifest mismatch must refuse")
    except RuntimeError as e:
        assert "MISMATCH" in str(e)
    print("[phase3] [smoke] resume MISMATCH branch: refused as designed")

    # degenerate-gate probes (data-dependent branches, outside the main leg):
    # (a) sketch-shaped (non-square) J refused at prediction_legs;
    torch.save({"J": torch.randn(5, hidden, generator=g)}, out / "j_sketch.pt")
    args2 = argparse.Namespace(**vars(args))
    args2.jlast = out / "j_sketch.pt"
    bank, _ = load_directions(args, hidden)
    try:
        prediction_legs(args2, bank, hidden)
        raise AssertionError("non-square J must be refused")
    except AssertionError as e:
        assert "FULL-RANK" in str(e)
    # (b) all-empty samples → row with n_kept_capture=0 and None reads (cell
    #     excluded downstream; capture is never called on an all-empty context).
    rows = cell_rows(
        [{"context_id": "cx", "source": "smoke", "system": None, "user": "q"}],
        [["", ""]],
        {},
        {"cx": []},
        {"cx": torch.zeros(hidden, dtype=torch.float64)},
        torch.zeros(2, hidden),
        direction="evil",
        alpha=1.0,
        mode="prefill",
    )
    assert rows[0]["n_kept_capture"] == 0 and rows[0]["cos_pred_jlast"] is None
    assert rows[0]["coherence_pass"] is False
    print("[phase3] [smoke] degenerate gates: non-square-J refusal + all-empty cell row")

    # Hook-parity probe (plan §8 risk row 2 + §12 assumptions 18/19; review v1
    # Major 2): with DeltaHook armed at layer L = source_layer, the block-L
    # OUTPUT (the cx_last capture slot: blocks[L] output = hidden_states[L+1]
    # values) shifts by EXACTLY alpha*delta at position T-1 and nowhere else;
    # block L-1 is byte-untouched; the downstream readout block moves at T-1
    # only (causality). Measurement note (found BY this probe, transformers
    # 4.57.6): the output_hidden_states recorder captures each block's output
    # BEFORE later-registered user forward hooks mutate it, so the post-edit
    # value must be read by a capture hook registered AFTER the DeltaHook —
    # the recorder tuple at the EDIT layer keeps the pre-hook value while
    # every DOWNSTREAM entry carries the propagated edit. Production is
    # unaffected (all phase-3 captures run UNHOOKED); the probe binds the
    # MODULE identity + position: DeltaHook edits exactly the tensor the
    # cx_last(L) convention reads, at exactly T-1.
    model, tok = load_model(args)  # fresh tiny model (fp32 CPU)
    ids = tok("What is the capital of France?", return_tensors="pt")["input_ids"]
    t_len = int(ids.shape[1])
    with torch.no_grad():
        ref_hs = model(ids, output_hidden_states=True).hidden_states
    gp = torch.Generator().manual_seed(11)
    probe_delta = torch.randn(hidden, generator=gp)
    probe_alpha = 3.0
    ls, lr = args.source_layer, args.readout_layer
    blocks = (model.model if hasattr(model, "model") else model).layers
    captured: dict[str, torch.Tensor] = {}

    def _cap(name):
        def _hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach().clone()

        return _hook

    with DeltaHook(model, ls, probe_delta, probe_alpha, expected_prompt_len=t_len) as hook:
        # Registered AFTER DeltaHook -> receives the post-edit block output.
        h_src = blocks[ls].register_forward_hook(_cap("src"))
        h_up = blocks[ls - 1].register_forward_hook(_cap("up"))
        try:
            with torch.no_grad():
                hook_hs = model(ids, output_hidden_states=True).hidden_states
        finally:
            h_src.remove()
            h_up.remove()
    assert hook.n_edits == 1, hook.n_edits
    # (a) exact edit at the capture slot: post-edit blocks[L] output == ref
    #     hidden_states[L+1] + alpha*delta at T-1, byte-identical elsewhere.
    d_src = (captured["src"] - ref_hs[ls + 1])[0]
    want = (probe_alpha * probe_delta).to(d_src.dtype)
    assert torch.allclose(d_src[t_len - 1], want, atol=1e-4), float(
        (d_src[t_len - 1] - want).abs().max()
    )
    assert d_src[: t_len - 1].abs().max().item() < 1e-8, "edit leaked off T-1 at the edit layer"
    # (b) upstream block L-1 byte-untouched — this ALSO re-verifies the slot
    #     convention in passing: blocks[L-1] output == hidden_states[L] values.
    assert torch.equal(captured["up"], ref_hs[ls]), "upstream block (L-1) output moved"
    # (c) downstream readout moved at T-1 ONLY (causality).
    d_ro = (hook_hs[lr + 1] - ref_hs[lr + 1])[0]
    assert d_ro[t_len - 1].abs().max().item() > 1e-6, "downstream readout did not move"
    assert d_ro[: t_len - 1].abs().max().item() < 1e-8, "causality broken (edit moved p < T-1)"
    recorder_sees_edit = not torch.equal(hook_hs[ls + 1], ref_hs[ls + 1])
    print(
        "[phase3] [smoke] hook-parity probe: blocks[L] output (cx_last slot) shifted by "
        f"exactly alpha*delta at T-1; upstream byte-equal; readout moved at T-1 only; "
        f"output_hidden_states recorder sees the edit at L: {recorder_sees_edit} "
        "(False on transformers 4.57.6 — pre-hook recording; captures run unhooked in prod)"
    )
    print("[phase3] [phase=smoke_done] PASS", flush=True)
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="#1776 Phase 3 steered-regeneration ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=["run", "smoke"], default="run")
    ap.add_argument("--contexts", type=Path, help="JSONL contexts / pair-manifest rows")
    ap.add_argument("--rb-dir", type=Path, help="dir with r_b {trait}.pt stacks")
    ap.add_argument("--mprime-weights", type=Path, help="m_ridge_x50k.pt payload")
    ap.add_argument("--jlast", type=Path, help="merged full-rank J_last.pt")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--eval-out", type=Path, default=None, help="phase3 eval JSON dest")
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--source-layer", type=int, default=C76.SOURCE_LAYER)
    ap.add_argument("--readout-layer", type=int, default=C76.READOUT_LAYER)
    ap.add_argument("--alphas", default="0.5,1,2,4")
    ap.add_argument("--k-samples", type=int, default=5)
    ap.add_argument("--k-baseline", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--directions", default=",".join(DEFAULT_DIRECTIONS))
    ap.add_argument("--random-seed", type=int, default=1776)
    ap.add_argument("--limit-contexts", type=int, default=0)
    ap.add_argument("--all-positions-subset", type=int, default=0, help="0 = off (exploratory)")
    ap.add_argument("--all-positions-alphas", default="4")
    ap.add_argument("--baseline-only", action="store_true", help="run baseline stratum then exit")
    ap.add_argument("--strata-shard", type=int, default=0, help="steered-strata shard index")
    ap.add_argument("--strata-num-shards", type=int, default=1, help="per-GPU strata fan-out")
    ap.add_argument("--finalize-only", action="store_true", help="aggregate a sharded out_root")
    args = ap.parse_args(argv)
    args.alphas = [float(x) for x in str(args.alphas).split(",") if x]
    args.all_positions_alphas = [float(x) for x in str(args.all_positions_alphas).split(",") if x]
    args.directions = [d for d in str(args.directions).split(",") if d]
    if args.eval_out is None:
        args.eval_out = args.out_root / "eval_results" / "issue_1776" / "phase3"
    if args.eval_out.suffix == ".json":
        # Crash-fix r9: a pre-fix dispatch passed the steered_shift_summaries.json
        # FILE path here; mkdir(parents=True) then misnested both deliverables one
        # level deep and the upload crashed at CommitOperationAdd ("not a file").
        ap.error(
            "--eval-out must be the eval DIRECTORY (deliverable FILEs are written "
            f"inside it); got file-shaped path: {args.eval_out}"
        )
    if args.mode == "smoke":
        return smoke(args)
    if args.finalize_only:
        return finalize_only(args)
    for req in ("contexts", "rb_dir", "mprime_weights", "jlast"):
        assert getattr(args, req) is not None, f"--{req.replace('_', '-')} is required for run"
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
