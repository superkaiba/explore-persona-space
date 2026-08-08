#!/usr/bin/env python3
"""#1902 steer_probe: representation-space intervention probe on OLMo-2 base.

Adds a fixed direction vector to the residual-stream hidden states (decoder
block outputs) at chosen token positions/layers during TEACHER-FORCED forwards
over the parent capture's plain render (``User: {q}\\nAssistant: {a}``, token-id
concatenation at the seam, offset-mapping spans — the issue1902_run recipe),
and reads the layer-31 pooled (mean over answer tokens) answer state per row,
matching the parent pooling exactly (fp32 accumulators over bf16 states).

Arms (one batched pass over the SAME batches per arm; shifts are vs baseline):
  baseline                no intervention
  rig_sanity_dy_ans       + dy at ANSWER positions at the CAPTURE layer —
                          expected pooled shift == dy (validates hooks+pooling)
  pre_L{16,24,28}_ctx     + v_pre (strong-band preimage of c*) at CONTEXT
                          positions of layer ell
  rand_L24_ctx            + norm-matched fixed-seed random vector (null)

Null-band mode (--null-band 'seeds=1903,...;layers=16,24'): the preimage/random
science arms are REPLACED by seeds x layers norm-matched (||v|| = ||v_pre||)
fixed-seed random arms (same RNG recipe as issue1902_steer_vectors.py);
baseline + rig-sanity are kept. --reuse-baseline stages the prior round's
committed baseline arm npz from HF outputs (resume-hit reuse), and
--baseline-recheck-rows N cross-checks it with a fresh unsteered forward
(per-row cos > 0.9999 on the first N rows; rc=4 on failure = staging bug).

GATE: rig-sanity must reproduce dy (cos > 0.99, norm ratio 0.95-1.05) or the
summary is marked FAIL and the process exits rc=3 (summary written first —
route on the artifact, not the rc).

Inputs (vectors + probe rows) stage from
hf:issue1902_stage_map/steer_probe/inputs/ when absent locally (local-first,
fail-loud). Outputs: per-arm fp16 npz under <out-root>/arms/ + steer_probe.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1902_common as C  # noqa: E402
import issue1902_run as R  # noqa: E402

INPUTS_HF_DIR = f"{C.HF_PREFIX}/steer_probe/inputs"
INPUT_FILES = (
    "probe_inputs.jsonl",
    "c_star.npy",
    "dy.npy",
    "v_pre.npy",
    "v_rand.npy",
    "meta.json",
)
RIG_COS_MIN = 0.99
RIG_NORM_RATIO = (0.95, 1.05)
OUTPUTS_HF_DIR = f"{C.HF_PREFIX}/steer_probe/outputs"
BASELINE_RECHECK_COS = 0.9999


def parse_null_band(spec: str) -> dict:
    """Parse 'seeds=1903,1904;layers=16,24' into {'seeds': [...], 'layers': [...]}."""
    parts = dict(kv.split("=", 1) for kv in spec.split(";") if kv.strip())
    out = {k: [int(x) for x in parts[k].split(",") if x.strip()] for k in ("seeds", "layers")}
    assert out["seeds"] and out["layers"], f"empty null-band spec: {spec!r}"
    assert len(set(out["seeds"])) == len(out["seeds"]), f"duplicate null-band seeds: {spec!r}"
    return out


def null_vector(seed: int, dim: int, norm: float) -> np.ndarray:
    """Fixed-seed random direction scaled to `norm` (issue1902_steer_vectors recipe)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v * (norm / np.linalg.norm(v))).astype(np.float32)


def stage_inputs(inputs_dir: Path) -> None:
    """Local-first staging: fetch any missing input from the HF data repo."""
    from explore_persona_space.orchestrate import hub

    inputs_dir.mkdir(parents=True, exist_ok=True)
    for name in INPUT_FILES:
        target = inputs_dir / name
        if not target.exists():
            hub.stage_hub_file(C.HF_DATA_REPO, f"{INPUTS_HF_DIR}/{name}", target)
            print(f"[steer] staged {name} <- hf:{INPUTS_HF_DIR}", flush=True)
    missing = [n for n in INPUT_FILES if not (inputs_dir / n).exists()]
    if missing:
        raise RuntimeError(f"steer_probe inputs missing after staging: {missing}")
    # Integrity: staged bytes must match the shas recorded at upload (meta.json is the
    # record itself, excluded) — a between-rounds re-upload of inputs/ would otherwise
    # silently change vectors/rows under the matched-target claim.
    import hashlib

    shas = json.load(open(inputs_dir / "meta.json")).get("files_sha256", {})
    for name in INPUT_FILES:
        if name == "meta.json" or name not in shas:
            continue
        got = hashlib.sha256((inputs_dir / name).read_bytes()).hexdigest()
        if got != shas[name]:
            raise RuntimeError(
                f"steer_probe input sha mismatch: {name} staged={got[:12]} recorded={shas[name][:12]}"
            )


def _prep_batch(entries: list[dict], device: str):
    """Pad + masks, parity with issue1902_run._pool_batch (right pad, pad_id 0)."""
    import torch

    bsz = len(entries)
    max_t = max(e["n_total"] for e in entries)
    ids = torch.full((bsz, max_t), 0, dtype=torch.long)
    mask = torch.zeros((bsz, max_t), dtype=torch.long)
    ctx_mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    ans_mask = torch.zeros((bsz, max_t), dtype=torch.bool)
    for b, e in enumerate(entries):
        seq = e["prompt_ids"] + e["answer_ids"]  # token-id concat (#1092)
        n_p, n_all = len(e["prompt_ids"]), len(seq)
        ids[b, :n_all] = torch.tensor(seq, dtype=torch.long)
        mask[b, :n_all] = 1
        ctx_mask[b, :n_p] = True  # context = full prompt (plan §4 P3)
        ans_mask[b, n_p:n_all] = True
    dev = torch.device(device)
    return ids.to(dev), mask.to(dev), ctx_mask.to(dev), ans_mask.to(dev)


def pool_batch_steered(
    model,
    blocks,
    entries: list[dict],
    *,
    capture_layer: int,
    steer_vec,
    steer_layer: int | None,
    steer_positions: str | None,
    device: str,
):
    """ONE batched teacher-forced forward with an optional additive hook on
    decoder block ``steer_layer``; returns the capture-layer answer-pooled
    states ``(B, H)`` (fp32 pooling accumulators, parity with _pool_batch)."""
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations

    ids, mask, ctx_mask, ans_mask = _prep_batch(entries, device)
    handles = []
    if steer_vec is not None:
        steer_mask = ctx_mask if steer_positions == "ctx" else ans_mask

        def _steer_hook(_module, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            h[steer_mask] += steer_vec.to(h.dtype)  # in-place: propagates onward

        handles.append(blocks[steer_layer].register_forward_hook(_steer_hook))
    try:
        captured = extract_layer_activations(model, ids, [capture_layer], attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    hs = captured[capture_layer].float()  # (B, T, H) fp32 accumulators
    ansf = ans_mask.to(torch.float32)
    pooled = (hs * ansf[..., None]).sum(1) / ansf.sum(1).clamp(min=1.0)[:, None]
    captured.clear()
    return pooled.cpu()


def run_arm(
    model,
    blocks,
    batches,
    inv,
    row_ids: list[str],
    *,
    arm_name: str,
    arm_idx: int,
    n_arms: int,
    npz_path: Path,
    capture_layer: int,
    steer_vec,
    steer_layer: int | None,
    steer_positions: str | None,
    device: str,
) -> np.ndarray:
    """Run (or resume) one arm; persist fp16 states the moment it completes."""
    import torch

    n_rows = len(row_ids)
    if npz_path.exists():
        d = np.load(npz_path, allow_pickle=False)
        if d["states"].shape[0] == n_rows and list(d["row_ids"]) == row_ids:
            print(f"[steer] arm {arm_idx}/{n_arms} {arm_name}: resume hit — skipped", flush=True)
            return d["states"].astype(np.float32)
        raise RuntimeError(f"stale arm npz {npz_path} (shape/id mismatch) — remove to recompute")
    t0 = time.time()
    chunks = []
    done = 0
    for batch in batches:
        chunks.append(
            pool_batch_steered(
                model,
                blocks,
                batch,
                capture_layer=capture_layer,
                steer_vec=steer_vec,
                steer_layer=steer_layer,
                steer_positions=steer_positions,
                device=device,
            )
        )
        done += len(batch)
        print(
            f"[steer] arm {arm_idx}/{n_arms} {arm_name} rows {done}/{n_rows} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    states = torch.cat(chunks, dim=0)[inv].to(torch.float16).numpy()
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")  # np.savez appends .npz to bare names
    np.savez(tmp, states=states, row_ids=np.asarray(row_ids))
    os.replace(tmp, npz_path)
    return states.astype(np.float32)


def baseline_recheck(
    model,
    blocks,
    entries: list[dict],
    baseline_states: np.ndarray,
    *,
    n: int,
    capture_layer: int,
    device: str,
) -> dict:
    """Fresh unsteered forward over the first n rows vs the resolved baseline states.

    Guards committed-baseline reuse across pods (same rows, same render, same
    pooling): per-row cos <= 0.9999 is a staging/render bug -> rc=4 fail-loud."""
    fresh = (
        pool_batch_steered(
            model,
            blocks,
            entries[:n],
            capture_layer=capture_layer,
            steer_vec=None,
            steer_layer=None,
            steer_positions=None,
            device=device,
        )
        .numpy()
        .astype(np.float32)
    )
    ref = baseline_states[:n].astype(np.float32)
    cos = (fresh * ref).sum(1) / (
        np.linalg.norm(fresh, axis=1) * np.linalg.norm(ref, axis=1) + 1e-12
    )
    stats = {"n_rows": int(n), "cos_min": float(cos.min()), "cos_mean": float(cos.mean())}
    ok = stats["cos_min"] > BASELINE_RECHECK_COS
    print(
        f"[steer] BASELINE-RECHECK {'PASS' if ok else 'FAIL'}: {stats} "
        f"(gate: cos_min>{BASELINE_RECHECK_COS})",
        flush=True,
    )
    if not ok:
        sys.exit(4)
    return stats


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _row_cos(rows: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (rows @ v) / (np.linalg.norm(rows, axis=1) * np.linalg.norm(v) + 1e-12)


def arm_summary(shift_rows: np.ndarray, c_star: np.ndarray, dy: np.ndarray) -> dict:
    mean_shift = shift_rows.mean(0)
    deciles = list(np.arange(0, 101, 10))
    return {
        "n_rows": int(shift_rows.shape[0]),
        "norm_mean_shift": float(np.linalg.norm(mean_shift)),
        "cos_mean_shift_c_star": _cos(mean_shift, c_star),
        "cos_mean_shift_dy": _cos(mean_shift, dy),
        "mean_row_shift_norm": float(np.linalg.norm(shift_rows, axis=1).mean()),
        "row_cos_c_star_deciles": [
            round(float(x), 4) for x in np.percentile(_row_cos(shift_rows, c_star), deciles)
        ],
        "row_cos_dy_deciles": [
            round(float(x), 4) for x in np.percentile(_row_cos(shift_rows, dy), deciles)
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs-dir", type=Path, default=Path("/workspace/issue1902_steer/inputs"))
    ap.add_argument("--out-root", type=Path, default=Path("/workspace/issue1902_steer"))
    ap.add_argument("--model-id", default=C.MODEL_IDS["B"])
    ap.add_argument("--revision", default="pin", help="'pin' = B pin from meta.json; 'none' = None")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=0, help="0 = all probe rows")
    ap.add_argument("--layers-sci", default="16,24,28", help="preimage-arm steer layers")
    ap.add_argument("--layer-rand", type=int, default=24)
    ap.add_argument("--layer-capture", type=int, default=31)
    ap.add_argument("--stage-only", action="store_true", help="stage inputs then exit")
    ap.add_argument(
        "--null-band",
        default="",
        help="'seeds=1903,1904;layers=16,24' — replace the preimage/random science arms "
        "with norm-matched (||v_pre||) fixed-seed random null arms",
    )
    ap.add_argument(
        "--reuse-baseline",
        action="store_true",
        help="stage the prior round's committed baseline arm npz from HF outputs "
        "(resume-hit reuse)",
    )
    ap.add_argument(
        "--baseline-recheck-rows",
        type=int,
        default=0,
        help="N>0: fresh unsteered forward over the first N rows; require per-row "
        "cos > 0.9999 vs the resolved baseline states (rc=4 on failure)",
    )
    ap.add_argument(
        "--upload",
        action="store_true",
        help="upload this run's NEW null-arm npzs + summary to the HF outputs dir "
        "(one commit, exact-set verified); committed prior-round arms are never overwritten",
    )
    args = ap.parse_args()

    import torch

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    stage_inputs(args.inputs_dir)
    if args.stage_only:
        print("[steer] --stage-only: inputs staged OK", flush=True)
        sys.exit(0)

    meta = json.load(open(args.inputs_dir / "meta.json"))
    revision = (
        None
        if args.revision == "none"
        else (meta["revision_pins"]["B"] if args.revision == "pin" else args.revision)
    )
    vecs = {n: np.load(args.inputs_dir / f"{n}.npy") for n in ("c_star", "dy", "v_pre", "v_rand")}
    rows = R._read_jsonl(args.inputs_dir / "probe_inputs.jsonl")
    if args.rows:
        rows = rows[: args.rows]
    row_ids = [r["id"] for r in rows]
    print(
        f"[steer] model={args.model_id} rev={str(revision)[:10]} rows={len(rows)} "
        f"device={args.device}",
        flush=True,
    )

    tokenizer = R._tokenizer(args.model_id, revision)
    model = R._load_hf_model(args.model_id, revision, args.device)
    hidden = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers)
    for n, v in vecs.items():
        assert v.shape == (hidden,), f"vector {n} shape {v.shape} != hidden ({hidden},)"

    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    blocks, _embed, _depth = _resolve_decoder_blocks(model)
    assert blocks is not None, "decoder blocks unresolvable — steering hooks need the hook path"
    layers_sci = [int(x) for x in args.layers_sci.split(",") if x.strip()]
    null_spec = parse_null_band(args.null_band) if args.null_band else None
    null_layers = null_spec["layers"] if null_spec else []
    for layer in [*layers_sci, args.layer_rand, args.layer_capture, *null_layers]:
        assert 0 <= layer < n_layers, f"layer {layer} out of range (n_layers={n_layers})"

    dev = torch.device(args.device)
    vt = {n: torch.tensor(v, dtype=torch.float32, device=dev) for n, v in vecs.items()}

    # Rig-sanity FIRST after baseline: gate before the science arms are trusted.
    arms: list[tuple[str, object, int | None, str | None]] = [
        ("baseline", None, None, None),
        (f"rig_sanity_dy_L{args.layer_capture}_ans", vt["dy"], args.layer_capture, "answer"),
    ]
    norm_target = float(np.linalg.norm(vecs["v_pre"]))
    if null_spec:
        for seed in null_spec["seeds"]:
            nv = torch.tensor(
                null_vector(seed, hidden, norm_target), dtype=torch.float32, device=dev
            )
            arms += [(f"null_s{seed}_L{ell}_ctx", nv, ell, "ctx") for ell in null_spec["layers"]]
    else:
        arms += [(f"pre_L{ell}_ctx", vt["v_pre"], ell, "ctx") for ell in layers_sci]
        arms += [(f"rand_L{args.layer_rand}_ctx", vt["v_rand"], args.layer_rand, "ctx")]

    if args.reuse_baseline:
        baseline_npz = args.out_root / "arms" / "baseline.npz"
        if not baseline_npz.exists():
            from explore_persona_space.orchestrate import hub

            hub.stage_hub_file(C.HF_DATA_REPO, f"{OUTPUTS_HF_DIR}/arms/baseline.npz", baseline_npz)
            print("[steer] staged committed baseline.npz <- hf outputs (resume reuse)", flush=True)

    entries = [
        R._capture_row_entry(
            tokenizer, {**r, "prefix_turns": None}, r["answer_text"], render="plain"
        )
        for r in rows
    ]
    for pos, e in enumerate(entries):
        e["_pos"] = pos
    batches = R._batches_by_token_budget(entries)
    inv = R._inverse_batch_order(batches, n_entries=len(entries))
    print(f"[steer] {len(entries)} rows -> {len(batches)} batches (arms={len(arms)})", flush=True)

    out_states: dict[str, np.ndarray] = {}
    rig_gate: dict | None = None
    recheck_stats: dict | None = None
    t_run = time.time()
    for k, (name, vec, layer, positions) in enumerate(arms, start=1):
        out_states[name] = run_arm(
            model,
            blocks,
            batches,
            inv,
            row_ids,
            arm_name=name,
            arm_idx=k,
            n_arms=len(arms),
            npz_path=args.out_root / "arms" / f"{name}.npz",
            capture_layer=args.layer_capture,
            steer_vec=vec,
            steer_layer=layer,
            steer_positions=positions,
            device=args.device,
        )
        if name == "baseline" and args.baseline_recheck_rows > 0:
            recheck_stats = baseline_recheck(
                model,
                blocks,
                entries,
                out_states["baseline"],
                n=args.baseline_recheck_rows,
                capture_layer=args.layer_capture,
                device=args.device,
            )
        if name.startswith("rig_sanity"):
            shift = out_states[name] - out_states["baseline"]
            mean_shift = shift.mean(0)
            dy = vecs["dy"]
            cos = _cos(mean_shift, dy)
            ratio = float(np.linalg.norm(mean_shift) / np.linalg.norm(dy))
            ok = cos > RIG_COS_MIN and RIG_NORM_RATIO[0] <= ratio <= RIG_NORM_RATIO[1]
            rig_gate = {"cos_vs_dy": cos, "norm_ratio": ratio, "pass": bool(ok)}
            print(
                f"[steer] RIG-SANITY {'PASS' if ok else 'FAIL'}: cos={cos:.5f} "
                f"norm_ratio={ratio:.4f} (gate: cos>{RIG_COS_MIN}, "
                f"ratio in {RIG_NORM_RATIO})",
                flush=True,
            )

    assert rig_gate is not None
    baseline = out_states["baseline"]
    summary = {
        "round": (
            "steer_probe null_band leg (inline scope-completion, task #1902)"
            if null_spec
            else "steer_probe (user-chat inline override, task #1902)"
        ),
        "model_id": args.model_id,
        "revision": revision,
        "device": args.device,
        "layer_capture": args.layer_capture,
        "n_rows": len(rows),
        "rig_sanity_gate": rig_gate,
        "baseline_recheck": recheck_stats,
        "predicted_reachable_magnitude": meta["vector_stats"]["predicted_reachable_magnitude"],
        "vector_norms": {n: float(np.linalg.norm(v)) for n, v in vecs.items()},
        "arms": {
            name: arm_summary(out_states[name] - baseline, vecs["c_star"], vecs["dy"])
            for name in out_states
            if name != "baseline"
        },
        "wall_s": round(time.time() - t_run, 1),
        "inputs_meta_sha": meta["files_sha256"],
        "metadata": {
            **as_metadata_dict(git_provenance()),
            "script": "scripts/issue1902_steer_probe.py",
        },
    }
    if null_spec:
        bands: dict[str, dict] = {}
        for ell in null_spec["layers"]:
            names = [f"null_s{s}_L{ell}_ctx" for s in null_spec["seeds"]]
            for key, field in (("c_star", "cos_mean_shift_c_star"), ("dy", "cos_mean_shift_dy")):
                vals = [summary["arms"][n][field] for n in names]
                bands.setdefault(str(ell), {})[key] = {
                    "values": [round(float(v), 6) for v in vals],
                    "mean": float(np.mean(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        summary["null_band"] = {
            "seeds": null_spec["seeds"],
            "layers": null_spec["layers"],
            "norm_target": norm_target,
            "bands": bands,
        }
    out_json = args.out_root / ("steer_probe_null_band.json" if null_spec else "steer_probe.json")
    R._write_json_atomic(out_json, summary)
    for name, s in summary["arms"].items():
        print(
            f"[steer] {name}: cos(c*)={s['cos_mean_shift_c_star']:+.4f} "
            f"cos(dy)={s['cos_mean_shift_dy']:+.4f} |mean_shift|={s['norm_mean_shift']:.3f}",
            flush=True,
        )
    if args.upload:
        assert null_spec, "--upload is wired for the null-band leg only (new files, no overwrite)"
        from explore_persona_space.orchestrate import hub

        api = R._hf_api()
        new_rel = sorted(f"arms/{name}.npz" for name, *_ in arms if name.startswith("null_"))
        new_rel.append(out_json.name)
        # allow_patterns restricts the commit to THIS run's new files — the committed
        # prior-round arms (baseline/pre_*/rand_*/rig_sanity) are never re-uploaded.
        hub.assert_hub_dir_filecounts(  # deterministic guard, outside the retry wrapper
            str(args.out_root),
            OUTPUTS_HF_DIR,
            allow_patterns=["arms/null_s*_ctx.npz", out_json.name],
        )
        hub.retry_transient(
            lambda: api.upload_folder(
                folder_path=str(args.out_root),
                path_in_repo=OUTPUTS_HF_DIR,
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                allow_patterns=["arms/null_s*_ctx.npz", out_json.name],
                commit_message="issue1902 steer_probe: null-band arms",
            ),
            what=f"upload_folder {OUTPUTS_HF_DIR} (null band)",
        )
        expected = [f"{OUTPUTS_HF_DIR}/{p}" for p in new_rel]
        missing = hub.verify_repo_paths_uploaded(
            api, C.HF_DATA_REPO, expected, path_in_repo=OUTPUTS_HF_DIR, repo_type="dataset"
        )
        if missing:
            raise RuntimeError(f"null-band outputs upload verify FAILED, missing: {missing}")
        print(f"[steer] uploaded + verified {len(expected)} files -> {OUTPUTS_HF_DIR}", flush=True)
    print(f"[steer] done: {out_json} (rig_sanity pass={rig_gate['pass']})", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0 if rig_gate["pass"] else 3)


if __name__ == "__main__":
    main()
