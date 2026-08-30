#!/usr/bin/env python3
"""Issue #2552 exactrep follow-up — assemble paired answer/context stores and train
flat Der-recipe SAEs on either object (thin driver over the #2476 trainer kernels).

Recipe pinned to arXiv 2606.28548 App. A via the parent's constants (identical to the
#2552 P1.2 replication SAE): BatchTopK width 32,768, k=128, flat tier bounds (32,768,),
lr 2e-4, batch 256, 3 epochs, Adam (0.9, 0.999), threshold EMA 0.999, seed 2552.
Trainer kernels are IMPORTED from the main-resident `scripts/issue2476_turnavg_sae.py`
(MatryoshkaBatchTopKSAE, _block_batches, _recon_fve, SAE_* constants) — verified
byte-identical, for these kernels, to the #2552 vendoring pin d8e9f8bdd4 (the branch
diff touches other functions only). The train loop mirrors the branch-only
`issue2552_turnsae_der.phase_sae_train` (b_dec init = seeded train-subsample mean,
per-epoch checkpoint + resume, epoch-end val FVE) with the #2476 split machinery
replaced by a self-contained carve of the NEW store.

Phases:
  assemble  concatenate answer chunks (chunk_*.npy) or paired context chunks
            (context_*.npy), selected by --vector-kind, with shared rows.jsonl
            into Y19.fp16.npy (memmap) + row_index.jsonl + assemble.done.json
  split     seeded (2552) carve: holdout 20,000 + val 10,000, train = rest
            (paper-underspecified; sizes mirror the parent's carve — a stated
            assumption; scaled down with floors at smoke n)
  train     the Der-recipe loop; ckpt_last.pt per epoch (resume; weights_only=False —
            self-produced bundle); final holdout var-FVE/nMSE report. The paper's
            0.097 reference applies only to the answer object, never to context states.
  all       assemble -> split -> train

Smoke = production with small dials (--steps-cap; realized-n split floors); the
production halt (FVE floor 0.5, parent G1 convention) stays byte-identical and is
enforced only under --production.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2476_turnavg_sae as T  # noqa: E402  (main-resident trainer kernels)
import issue2552_exactrep_prep as PREP  # noqa: E402  (_write_json_atomic)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2552_exactrep_train")

# Der recipe constants — same values as the branch driver's REP_* block (plan §11).
REP_DICT = 32_768
REP_K = 128
REP_TIER_BOUNDS = (32_768,)  # FLAT — never the matryoshka tier derivation
REP_SEED = 2552
HOLDOUT_N = 20_000
VAL_N = 10_000
G1_FVE_FLOOR = 0.5  # parent G1 halt floor (production only)
NMSE_ADVISORY_BAND = (0.07, 0.15)  # paper 0.097; parent realized 0.0778
PAPER_NMSE = 0.097
DEFAULT_RUN_ROOT = Path("/workspace/eps-2552-exactrep")


def derive_splits(n_rows: int, seed: int = REP_SEED) -> dict[str, np.ndarray]:
    """Seeded holdout/val/train carve with realized-n floors (smoke-safe arithmetic:
    dials derive from REALIZED n, never assumed caps)."""
    assert n_rows >= 10, f"store too small to split: {n_rows}"
    holdout_n = HOLDOUT_N if n_rows > 5 * HOLDOUT_N else max(2, n_rows // 10)
    val_n = VAL_N if n_rows > 5 * (HOLDOUT_N + VAL_N) else max(2, n_rows // 10)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    return {
        "holdout": np.sort(perm[:holdout_n]),
        "val": np.sort(perm[holdout_n : holdout_n + val_n]),
        "train": np.sort(perm[holdout_n + val_n :]),
    }


def _store_dirs(args) -> list[Path]:
    dirs = [Path(d) for d in args.store_dirs]
    for d in dirs:
        assert d.is_dir(), d
    return dirs


def _vector_chunks(store_dir: Path, vector_kind: str) -> list[Path]:
    pattern = "chunk_*.npy" if vector_kind == "answer" else "context_*.npy"
    return list(store_dir.glob(pattern))


def _answer_stem(path: Path, vector_kind: str) -> str:
    if vector_kind == "answer":
        return path.stem
    assert path.stem.startswith("context_"), path
    return f"chunk_{path.stem.removeprefix('context_')}"


def phase_assemble(args) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "assemble.done.json"
    y_path = out / "Y19.fp16.npy"
    idx_path = out / "row_index.jsonl"
    chunks: list[Path] = []
    for d in _store_dirs(args):
        chunks.extend(_vector_chunks(d, args.vector_kind))
    chunks = sorted(chunks, key=lambda p: p.name)
    assert chunks, f"no capture chunks under {args.store_dirs}"
    names = [c.name for c in chunks]
    assert len(names) == len(set(names)), "duplicate chunk gci across store dirs"
    if done.exists():
        doc = json.loads(done.read_text())
        if (
            doc.get("chunk_names") == names
            and doc.get("vector_kind") == args.vector_kind
            and y_path.exists()
            and idx_path.exists()
        ):
            logger.info("[assemble] resume: %d chunks already assembled; skip", len(names))
            return
    # completeness: every chunk needs its done sentinel (a mid-write chunk never assembles)
    for c in chunks:
        stem = _answer_stem(c, args.vector_kind)
        sent = c.with_name(f"{stem}.done.json")
        assert sent.exists(), f"chunk without done sentinel: {c}"
    sizes, hidden = [], None
    for c in chunks:
        arr = np.load(c, mmap_mode="r")
        assert arr.dtype == np.float16 and arr.ndim == 2, (c, arr.dtype, arr.shape)
        hidden = arr.shape[1] if hidden is None else hidden
        assert arr.shape[1] == hidden, (c, arr.shape, hidden)
        sizes.append(arr.shape[0])
    n_rows = int(sum(sizes))
    assert n_rows > 0, "assemble found zero rows"
    tmp_y = out / ".Y19.tmp.npy"
    mm = np.lib.format.open_memmap(tmp_y, mode="w+", dtype=np.float16, shape=(n_rows, hidden))
    tmp_idx = out / ".row_index.tmp.jsonl"
    cursor = 0
    with tmp_idx.open("w", encoding="utf-8") as f:
        for c, sz in zip(chunks, sizes, strict=True):
            mm[cursor : cursor + sz] = np.load(c, mmap_mode="r")
            stem = _answer_stem(c, args.vector_kind)
            rows_file = c.with_name(f"{stem}.rows.jsonl")
            k = 0
            with rows_file.open(encoding="utf-8") as rf:
                for line in rf:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    r["row"] = cursor + k
                    # Logical row identity is shared across the paired answer/context
                    # stores. Keep the answer-chunk name in BOTH row indices so they
                    # are byte-identical; assemble.done.json records vector_kind.
                    r["chunk"] = f"{stem}.npy"
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    k += 1
            assert k == sz, f"rows.jsonl/npy row-count mismatch for {c}: {k} vs {sz}"
            cursor += sz
    mm.flush()
    del mm
    tmp_y.replace(y_path)
    tmp_idx.replace(idx_path)
    PREP._write_json_atomic(
        done,
        {
            "n_rows": n_rows,
            "hidden": int(hidden),
            "chunk_names": names,
            "vector_kind": args.vector_kind,
            "metadata": as_metadata_dict(
                git_provenance(), phase=f"exactrep-assemble-{args.vector_kind}"
            ),
        },
    )
    print(
        f"[assemble] kind={args.vector_kind} {n_rows} rows x {hidden} from {len(chunks)} chunks",
        flush=True,
    )


def phase_train(args) -> None:
    out = Path(args.out_dir)
    y_path = out / "Y19.fp16.npy"
    assert y_path.exists(), "run --phase assemble first"
    y_mm = np.load(y_path, mmap_mode="r")
    n_rows, hidden = y_mm.shape
    if args.production:
        assert n_rows > 5 * (HOLDOUT_N + VAL_N), (
            f"--production needs the full store (got {n_rows} rows)"
        )
    splits = derive_splits(n_rows)
    np.savez(out / "splits.npz", **splits)
    pool_doc = {k: int(len(v)) for k, v in splits.items()} | {
        "n_rows": int(n_rows),
        "seed": REP_SEED,
    }
    print(f"[train] pools: {json.dumps(pool_doc)}", flush=True)

    dev = args.device
    model = T.MatryoshkaBatchTopKSAE(
        act_dim=int(hidden),
        dict_size=args.dict_size,
        k=REP_K,
        tier_bounds=(args.dict_size,),
        seed=REP_SEED,
    ).to(dev)
    assert model.tier_bounds == (args.dict_size,), "flat replication SAE must be 1-tier"
    tr_pos, val_pos = splits["train"], splits["val"]
    # b_dec init: seeded train-subsample mean (parent phase_sae_train convention)
    rng0 = np.random.default_rng(REP_SEED + 1)
    sub = np.sort(rng0.choice(tr_pos, size=min(65_536, len(tr_pos)), replace=False))
    mu = np.zeros(model.act_dim, dtype=np.float64)
    for s in range(0, len(sub), 8192):
        mu += np.asarray(y_mm[sub[s : s + 8192]], np.float64).sum(0)
    with torch.no_grad():
        model.b_dec.copy_(torch.as_tensor(mu / len(sub), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=T.SAE_LR, betas=T.SAE_ADAM_BETAS)

    ckpt_path = out / "ckpt_last.pt"
    start_epoch, step = 0, 0
    epoch_rows: list[dict] = []
    steps_cap = int(args.steps_cap)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)  # self-produced
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = int(ck["epoch_done"]), int(ck["step"])
        epoch_rows = list(ck["log_rows"])
        if bool(ck.get("steps_capped")) and steps_cap and step >= steps_cap:
            start_epoch = T.SAE_EPOCHS  # a steps-capped smoke resume never re-trains
        logger.info("[train] RESUMED at epoch %d (step %d)", start_epoch, step)
    t0 = time.time()
    stop = False
    for epoch in range(start_epoch, T.SAE_EPOCHS):
        rng_e = np.random.default_rng(REP_SEED * 1000 + epoch)
        run_loss, run_n = 0.0, 0
        diags: dict = {"l0_train": float("nan")}
        for xb in T._block_batches(y_mm, tr_pos, T.SAE_BATCH, rng_e):
            x = torch.as_tensor(np.asarray(xb, np.float32), device=dev)
            loss, diags, _fired = model.train_step_losses(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += diags["loss"]
            run_n += 1
            step += 1
            if step % 200 == 0:
                print(
                    f"[train] epoch {epoch + 1}/{T.SAE_EPOCHS} step {step} "
                    f"loss={run_loss / max(1, run_n):.1f} thr={float(model.threshold):.4f} "
                    f"l0={diags['l0_train']:.0f} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            if steps_cap and step >= steps_cap:
                stop = True
                break
        fve_val, l0_val = T._recon_fve(model, y_mm, val_pos)
        row = {
            "epoch": epoch + 1,
            "steps": step,
            "mean_loss": round(run_loss / max(1, run_n), 3),
            "val_var_fve": round(fve_val, 6),
            "val_nmse": round(1.0 - fve_val, 6),
            "val_l0": round(l0_val, 2),
            "threshold": float(model.threshold),
            "elapsed_s": round(time.time() - t0, 1),
        }
        epoch_rows.append(row)
        print(f"[train] epoch-done {json.dumps(row)}", flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch_done": epoch if stop else epoch + 1,
                "steps_capped": bool(stop),
                "step": step,
                "log_rows": epoch_rows,
            },
            ckpt_path,
        )
        if stop:
            break
    assert epoch_rows, "train produced no epoch rows"

    fve_hold, l0_hold = T._recon_fve(model, y_mm, splits["holdout"])
    nmse_hold = 1.0 - fve_hold
    is_paper_object = args.vector_kind == "answer"
    in_band = (
        NMSE_ADVISORY_BAND[0] <= nmse_hold <= NMSE_ADVISORY_BAND[1] if is_paper_object else None
    )
    model.save_dir(out)
    cfg_path = out / "cfg.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update(
        {
            "vector_kind": args.vector_kind,
            "training_object": (
                "assistant-content token mean"
                if is_paper_object
                else "pre-assistant generation-prompt last-token context state"
            ),
        }
    )
    PREP._write_json_atomic(cfg_path, cfg)
    report = {
        "pools": pool_doc,
        "epochs": epoch_rows,
        "holdout_var_fve": round(fve_hold, 6),
        "holdout_nmse": round(nmse_hold, 6),
        "holdout_l0": round(l0_hold, 2),
        "paper_nmse": PAPER_NMSE if is_paper_object else None,
        "nmse_advisory_band": list(NMSE_ADVISORY_BAND) if is_paper_object else None,
        "nmse_in_band": in_band,
        "recipe": {
            "dict_size": args.dict_size,
            "k": REP_K,
            "tier_bounds": [args.dict_size],
            "lr": T.SAE_LR,
            "batch": T.SAE_BATCH,
            "epochs": T.SAE_EPOCHS,
            "adam_betas": list(T.SAE_ADAM_BETAS),
            "threshold_ema": T.SAE_THRESH_EMA,
            "seed": REP_SEED,
        },
        "training_object": (
            "assistant-content token mean"
            if is_paper_object
            else "pre-assistant generation-prompt last-token context state"
        ),
        "vector_kind": args.vector_kind,
        "steps_cap": steps_cap,
        "metadata": as_metadata_dict(git_provenance(), phase="exactrep-train"),
    }
    PREP._write_json_atomic(out / "train_log.json", report)
    print(
        f"[train] done holdout_var_fve={fve_hold:.4f} nmse={nmse_hold:.4f} "
        f"(paper {PAPER_NMSE if is_paper_object else 'n/a'}) l0={l0_hold:.1f}",
        flush=True,
    )
    if is_paper_object and not in_band:
        logger.warning("[train] ADVISORY: nMSE %.4f outside band %s", nmse_hold, NMSE_ADVISORY_BAND)
    if args.production and fve_hold < G1_FVE_FLOOR:
        logger.error("[train] G1 FAIL: holdout FVE %.4f < %.2f", fve_hold, G1_FVE_FLOOR)
        raise SystemExit(25)  # RC_G1, parent convention


PHASES = {"assemble": phase_assemble, "train": phase_train}


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #2552 exactrep SAE retrain (Der recipe).")
    ap.add_argument("--phase", default="all", choices=["all", *PHASES])
    ap.add_argument(
        "--store-dirs",
        nargs="+",
        default=["/workspace/eps-2552-exactrep/store"],
        help="capture out-dirs (all shards' chunk files)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to sae_rep for answer vectors and sae_ctx_rep for context vectors",
    )
    ap.add_argument("--vector-kind", choices=["answer", "context"], default="answer")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dict-size", type=int, default=REP_DICT)
    ap.add_argument("--steps-cap", type=int, default=0, help="0 = full 3-epoch train")
    ap.add_argument("--production", action="store_true", help="enforce full-store + G1 floor")
    ap.add_argument("--import-check", action="store_true", help="argparse-attr completeness")
    ap.add_argument("--list-phases", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        return 0
    if args.list_phases:
        print(sorted(PHASES))
        return 0
    if args.out_dir is None:
        leaf = "sae_rep" if args.vector_kind == "answer" else "sae_ctx_rep"
        args.out_dir = DEFAULT_RUN_ROOT / leaf
    if args.production:
        assert args.dict_size == REP_DICT and not args.steps_cap, (
            "--production refuses smoke dials (dict-size/steps-cap)"
        )
    names = [args.phase] if args.phase != "all" else ["assemble", "train"]
    for name in names:
        PHASES[name](args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension teardown (rc race)
