"""#1776 Phase 0.3: teacher-forced parity rig (G-PARITY, ENGINEERING gate — blocks).

Re-runs teacher-forced capture for a seeded ~200-row sample of the #779 stored
pairs (staged capture chunks + their paired raw-completion jsons) and requires
per-row cosine >= 0.999 between recomputed and stored values of EVERY consumed
field — cx_last(14) (J/M' input slot), cx_last(19) (shipped-M reference input),
v_x(19) (targets); span-mean summaries, flat 0.999 bar (#779 pass-1 precedent
realized 0.999748; plan §7 G-PARITY). Rows < 0.999 go on the SHARED exclusion
list (excluded from BOTH arms everywhere, plan §3). Sample-level HALT rc=8
(+ report JSON) only if > ``--max-fail-frac`` (5%) of rows fail.

Capture convention is pinned to the PRODUCER: per-row forwards through the
parent's own ``capture_context_vector`` / ``capture_answer_vector``
(issue779_collect.py) — deliberately NOT batched: parity must reproduce the
exact producer convention (padding-free per-row forwards), and n<=200 rows x 2
forwards is minutes on the pod (vectorize-first carve-out: convention parity).

Per-unit persistence + resume (code-style T2: 200 units > 50): each row's
cosines append atomically to ``rows.jsonl`` keyed on (chunk, ci, model,
layers, dtype); a re-run skips completed keys.

Content hygiene: staged chunks carry real LMSYS/WildChat text — this rig NEVER
prints prompt/response text; logs carry chunk names, ci indices, cosines only.

Tiny-real CPU smoke: ``--tiny-self-test`` runs the FULL rig body on a
from-config tiny Qwen2 (real tokenizer): capture-once = "stored", re-capture =
"recomputed" (cosine 1.0 passes), then a perturbed stored row exercises the
exclusion branch AND the >5% sample-level HALT branch (degenerate-gate probe).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_collect as COL  # noqa: E402
import issue779_common as C  # noqa: E402


def consumed_fields(src_layer: int, ro_layer: int) -> tuple[tuple[str, int], ...]:
    """The three consumed (field, layer) slots (plan §7 G-PARITY): cx_last at the
    SOURCE layer (J/M' input), cx_last at the READOUT layer (shipped-M reference
    input), v_x at the READOUT layer (targets). Production: (14, 19); the tiny
    self-test scales the pair down to fit a 4-layer from-config model."""
    assert src_layer < ro_layer, (src_layer, ro_layer)
    return (("cx_last", src_layer), ("cx_last", ro_layer), ("v_x", ro_layer))


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.to(torch.float64).flatten()
    b64 = b.to(torch.float64).flatten()
    denom = float(a64.norm() * b64.norm())
    assert denom > 0, "zero-norm activation in parity comparison"
    return float((a64 @ b64) / denom)


def _row_key(chunk: str, ci: int, args) -> str:
    return f"{chunk}|ci={ci}|model={args.model}|L={args.source_layer},{args.readout_layer}"


def _load_done(rows_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    return done


def _append_row(rows_path: Path, row: dict) -> None:
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with rows_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def sample_rows(
    chunks_dir: Path, raw_dir: Path, n_rows: int, seed: int, fields: tuple
) -> list[dict]:
    """Seeded row sample across the staged chunk pairs.

    Yields {chunk, row_idx, ci, prompt, response, stored: {(field, layer): (H,)}}.
    Chunk .pt fields are (n, len(layers), H) with a ``layers`` list
    (issue779 _stack_chunk); responses join from the paired raw json by ci.
    """
    pt_files = sorted(chunks_dir.glob("shard*_chunk*.pt"))
    assert pt_files, f"no staged capture chunks under {chunks_dir}"
    rng = np.random.default_rng(seed)
    pool: list[dict] = []
    for pt in pt_files:
        bundle = torch.load(pt, map_location="cpu", weights_only=True)
        layers = [int(x) for x in bundle["layers"]]
        for _, li in fields:
            assert li in layers, f"{pt.name} layers={layers} missing consumed layer {li}"
        raw_path = raw_dir / (pt.stem + ".json")
        assert raw_path.exists(), f"paired raw json absent: {raw_path}"
        raw = json.loads(raw_path.read_text())
        resp_by_ci = {int(r["ci"]): r["response"] for r in raw["rows"]}
        for row_idx, ci in enumerate(int(c) for c in bundle["ci"]):
            assert ci in resp_by_ci, f"{pt.name} ci={ci} missing from raw json"
            stored = {
                (fld, li): bundle[fld][row_idx, layers.index(li), :].clone() for fld, li in fields
            }
            pool.append(
                {
                    "chunk": pt.name,
                    "ci": ci,
                    "prompt": bundle["prompts"][row_idx],
                    "response": resp_by_ci[ci],
                    "stored": stored,
                }
            )
    assert len(pool) >= n_rows, (
        f"only {len(pool)} rows staged but --n-rows={n_rows}; stage more parity chunks"
    )
    idx = rng.choice(len(pool), size=n_rows, replace=False)
    return [pool[i] for i in sorted(idx)]


def recompute_row(model, tok, prompt: str, response: str, fields: tuple) -> dict:
    """Teacher-forced recompute of the consumed fields via the PRODUCER's rig."""
    layers = sorted({li for _, li in fields})
    msgs = [{"role": "user", "content": prompt}]
    cx = COL.capture_context_vector(model, tok, msgs, layers)
    av = COL.capture_answer_vector(model, tok, msgs, response, layers, {}, keep_per_token=False)
    assert av is not None, "empty response in parity sample (producer kept non-empty only)"
    out = {}
    for fld, li in fields:
        src = cx["last"] if fld == "cx_last" else av["v_x"]
        out[(fld, li)] = src[layers.index(li), :]
    return out


def run_parity(rows: list[dict], model, tok, args) -> int:
    """Score each row; write per-row JSONL + the exclusion list + report. rc contract:
    0 = PASS (fail_frac <= max), 8 = sample-level HALT."""
    rows_path = args.out_dir / "rows.jsonl"
    done = _load_done(rows_path)
    results: list[dict] = []
    t0 = time.time()
    for k, row in enumerate(rows):
        key = _row_key(row["chunk"], row["ci"], args)
        if key in done:
            results.append(done[key])
            continue
        fields = consumed_fields(args.source_layer, args.readout_layer)
        rec = recompute_row(model, tok, row["prompt"], row["response"], fields)
        cosines = {
            f"{fld}_{li}": _cos(rec[(fld, li)], row["stored"][(fld, li)]) for fld, li in fields
        }
        rec_row = {
            "key": key,
            "chunk": row["chunk"],
            "ci": row["ci"],
            "cosines": cosines,
            "pass": bool(all(v >= args.threshold for v in cosines.values())),
        }
        _append_row(rows_path, rec_row)
        results.append(rec_row)
        print(
            f"[parity] unit {k + 1}/{len(rows)} {row['chunk']}:ci={row['ci']} "
            f"min_cos={min(cosines.values()):.6f} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    failed = [r for r in results if not r["pass"]]
    fail_frac = len(failed) / len(results)
    exclusion = [{"chunk": r["chunk"], "ci": r["ci"]} for r in failed]
    report = {
        "gate": "G-PARITY",
        "threshold": args.threshold,
        "n_rows": len(results),
        "n_failed": len(failed),
        "fail_frac": fail_frac,
        "max_fail_frac": args.max_fail_frac,
        "halt": bool(fail_frac > args.max_fail_frac),
        "min_cosine_overall": min(min(r["cosines"].values()) for r in results),
        "exclusion_list": exclusion,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(args.out_dir / "parity_report.json", report)
    C76.atomic_write_json(args.out_dir / "exclusion_list.json", {"excluded": exclusion})
    status = "HALT rc=8" if report["halt"] else "PASS"
    print(
        f"[parity] [phase=parity_done] {status} failed={len(failed)}/{len(results)} "
        f"-> {args.out_dir}/parity_report.json",
        flush=True,
    )
    return 8 if report["halt"] else 0


def tiny_self_test(args) -> int:
    """CPU smoke: full rig body on a tiny-real model + a forced-failure probe."""
    import issue1776_jlens_fit as JF

    _, model, tok = JF.load_lens_model(C.DEFAULT_MODEL, device="cpu", tiny=True)
    args.source_layer, args.readout_layer = 1, 3  # tiny 4-layer model
    fields = consumed_fields(args.source_layer, args.readout_layer)
    prompts = ["What is the capital of France?", "Name one prime number."]
    responses = ["The capital of France is Paris.", "Two is a prime number."]
    rows = []
    for i, (p, r) in enumerate(zip(prompts, responses, strict=True)):
        stored = recompute_row(model, tok, p, r, fields)  # capture-once = "stored"
        rows.append({"chunk": "tiny0.pt", "ci": i, "prompt": p, "response": r, "stored": stored})
    rc = run_parity(rows, model, tok, args)
    assert rc == 0, f"self-parity should PASS (got rc={rc})"
    rep = json.loads((args.out_dir / "parity_report.json").read_text())
    assert rep["min_cosine_overall"] >= args.threshold, rep["min_cosine_overall"]
    # Degenerate-gate probe: perturb one stored field -> exclusion + (1/2 > 5%) HALT.
    rows[0]["stored"][("v_x", args.readout_layer)] = torch.randn_like(
        rows[0]["stored"][("v_x", args.readout_layer)]
    )
    args.out_dir = args.out_dir / "halt_probe"
    rc2 = run_parity(rows, model, tok, args)
    assert rc2 == 8, f"perturbed row must trip the sample-level HALT (got rc={rc2})"
    rep2 = json.loads((args.out_dir / "parity_report.json").read_text())
    assert rep2["n_failed"] == 1 and rep2["exclusion_list"], rep2
    print("[parity] [phase=tiny_self_test_done] PASS (self-cos>=0.999; HALT branch fired)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument(
        "--chunks-dir",
        type=Path,
        default=C76.DATA_DIR
        / "hf_dl"
        / "issue779_monitoring"
        / "fitter-fair-comparison-n1m"
        / "final_token_capture",
    )
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=C76.DATA_DIR
        / "hf_dl"
        / "issue779_monitoring"
        / "fitter-fair-comparison-n1m"
        / "raw_completions",
    )
    ap.add_argument("--n-rows", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source-layer", type=int, default=C76.SOURCE_LAYER)
    ap.add_argument("--readout-layer", type=int, default=C76.READOUT_LAYER)
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--max-fail-frac", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, default=C76.DATA_DIR / "parity")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny-self-test", action="store_true", help="CPU smoke (no staged data)")
    args = ap.parse_args(argv)

    if args.tiny_self_test:
        return tiny_self_test(args)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    fields = consumed_fields(args.source_layer, args.readout_layer)
    rows = sample_rows(args.chunks_dir, args.raw_dir, args.n_rows, args.seed, fields)
    return run_parity(rows, model, tok, args)


if __name__ == "__main__":
    sys.exit(main())
