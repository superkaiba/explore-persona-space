"""Production-entrypoint smoke legs for issue #2587 (r2 concern `smoke-run-coverage`).

Two legs, each running the ACTUAL production entrypoint as a REAL subprocess
with a REAL exit code and a REAL artifact on disk:

- ``analysis`` — builds a fully-LOCAL tiny world at the REAL dims the spec
  asserts (H_9B=4096 / H_7B=3584; 9B store layers derived from a frozen
  L*=22 -> (16, 22, 30); parent layers exactly (14, 19, 26)) and runs
  ``scripts/issue2587_analysis.py --smoke`` end to end. ZERO fakes: every
  input resolves locally, so no network path is reachable. The child log is
  asserted to carry one ``[smoke-blind-spot]`` line per registered
  assert-skipped site (the r1 `analysis-smoke-blindspots` registry, FIRING
  through the production entrypoint).
- ``judge`` — builds anchors from the REAL sha-pinned bank2564 module
  (``judged_specs``/``lang_specs`` on the production slice) and runs
  ``scripts/issue2587_judge.py --smoke --skip-upload`` in a child process
  whose ONLY fake sits at the TRUE network boundary:
  ``eval.batch_judge.judge_completions_batch``, replaced with a
  ``create_autospec`` fake (signature-conformant by construction) that pins
  the plan invariants (judge model / max_tokens=1024 / threshold_base=0)
  and writes a real ``save_raw`` payload the REAL
  ``judge_result_from_save_raw`` reduce then consumes. ``judge_graded``'s
  real body (item packing, custom_id grammar, drop-class accounting) runs.
  The fake is installed by this driver's ``judge-child`` mode, which then
  ``runpy``-executes the judge script as ``__main__`` — so the child's exit
  code IS the production entrypoint's.

No pod, no GPU, no API calls. Artifacts land under ``--out-root``
(default ``/tmp/issue-2587-smoke/r2d1``). The tiny-world shape is imported
from ``tests/test_issue2587_analysis.py`` (single source of truth for the
fixture world; a deliberate scripts->tests coupling for this smoke driver).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps before torch import (code-style shared-VM rule)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import runpy  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TESTS = REPO_ROOT / "tests"
for _p in (SCRIPTS, TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_OUT_ROOT = Path("/tmp/issue-2587-smoke/r2d1")
LSTAR = 22  # in TWIN_LAYERS_9B -> 9B store layers (16, 22, 30)
REF7B_COMMIT = "smoke-fixture-2587"  # fixture provenance token (not a git sha)
SEED = 2587

# The six assert-skipped registry sites whose skip lines the analysis child
# log MUST carry (bootstrap_null_b is param-narrowed: warned from build_config).
ANALYSIS_SKIP_SITES = (
    "expected_contexts",
    "expected_pairs",
    "axis_completeness",
    "bank9_cardinality",
    "carrier_count_12",
    "parent_axes_count",
)


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ── analysis-leg tiny world (REAL dims; every input local) ─────────────


def _write_cell_stores(
    root: Path,
    sub_vc: str,
    sub_va: str,
    bank: dict,
    layers: tuple[int, ...],
    d: int,
    rng: np.random.Generator,
    *,
    rows_key: str,
    with_leak: bool,
) -> None:
    """Per-cell vc/va .pt stores in the unit-3b layout the loaders consume."""
    import issue2587_analysis as AN

    contexts = bank["contexts"]
    cells = sorted({c["cell"] for c in contexts.values()})
    conv = f"{AN.LAYER_CONVENTION_SUBSTR} (smoke fixture)"
    n_layers = len(layers)
    single_vc = sub_vc.endswith(".pt")  # 7B: ONE bank file, not per-cell
    if single_vc:
        cids = sorted(contexts)
        vc = torch.from_numpy(rng.standard_normal((len(cids), n_layers, d)).astype(np.float32))
        p = root / sub_vc
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"layers": list(layers), "layer_convention": conv, "vc": vc, "context_ids": cids}, p
        )
    for cell in cells:
        cids = sorted(cid for cid, c in contexts.items() if c["cell"] == cell)
        if not single_vc:
            vc = torch.from_numpy(rng.standard_normal((len(cids), n_layers, d)).astype(np.float32))
            p = root / sub_vc / f"{cell}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"layers": list(layers), "layer_convention": conv, "vc": vc, "context_ids": cids},
                p,
            )
        rows = []
        for cid in cids:
            for draw in range(4):
                row = {
                    "context_id": cid,
                    "draw": draw,
                    "n_completion_tokens": int(rng.integers(5, 40)),
                }
                if with_leak:
                    # ONE leaked row per cell (ctx0, draw 3): the think-leak
                    # exclusion path runs; ctx0 keeps 3 valid draws.
                    row["think_leak"] = cid == cids[0] and draw == 3
                rows.append(row)
        n_rows = len(rows)
        tail = torch.from_numpy(rng.standard_normal((n_rows, n_layers, d)).astype(np.float32))
        span = torch.from_numpy(rng.standard_normal((n_rows, n_layers, d)).astype(np.float32))
        # ONE empty row per cell (ctx1, draw 3) exercises the empty exclusion.
        empty_rows = [7] if len(cids) >= 2 else []
        name = f"va2564_{cell}.pt" if rows_key == "index" else f"{cell}.pt"
        p = root / sub_va / name
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layers": list(layers),
                "layer_convention": conv,
                rows_key: rows,
                "va_tail_incl": tail,
                "va_span": span,
                "empty_rows": empty_rows,
            },
            p,
        )


def _write_embeddings(root: Path, ctx_ids: list[str], rng: np.random.Generator, engine) -> None:
    p = root / "analysis_tensors" / "embeddings_qwen3_8b" / "means_anchors.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "context_ids": np.array(sorted(ctx_ids)),
        "emb_mean": rng.standard_normal((len(ctx_ids), 8)).astype(np.float32),
    }
    if engine is not None:
        arrays["vllm_version"] = np.array(engine)
    np.savez(p, **arrays)


def _mk_preds(rng: np.random.Generator, ids: list[str], layer: int, quality: float) -> dict:
    n, dd = len(ids), 8
    target = rng.standard_normal((n, dd))
    pred = quality * target + (1.0 - quality) * rng.standard_normal((n, dd))
    return {"layer": layer, "ci_te": list(ids), "pred_te": pred, "target_te": target}


def build_analysis_world(root: Path) -> list[str]:
    """Write the tiny world; return the analysis CLI argv tail (all local)."""
    import issue2587_analysis as AN
    import test_issue2587_analysis as TW  # tests/ tiny-world builders (module docstring)

    rng = _rng()
    root.mkdir(parents=True, exist_ok=True)
    bank9, bank7 = TW._banks()
    (root / "bank9.json").write_text(json.dumps(bank9))
    (root / "bank7.json").write_text(json.dumps(bank7))
    (root / "manip9.json").write_text(json.dumps(TW._fire_doc(pilots=True, not_fired="v3")))
    (root / "manip7.json").write_text(json.dumps(TW._fire_doc(pilots=False, not_fired="v2")))
    (root / "sweep.json").write_text(
        json.dumps({"lstar": {"frozen": True, "lstar": LSTAR, "note": "smoke fixture"}})
    )
    (root / "ref7b.json").write_text(
        json.dumps(
            {
                "axes": {f"refax{i:02d}": {} for i in range(11)},
                "contract": {},
                "meta": {"note": "smoke fixture — 11 dummy parent axes (load_ref7b_parent)"},
            }
        )
    )

    layers9 = tuple(sorted({LSTAR, *AN.TWIN_LAYERS_9B}))
    root9 = root / "root9"
    root7 = root / "root7"
    _write_cell_stores(
        root9,
        "analysis_tensors/vc2587",
        "analysis_tensors/va2587",
        bank9,
        layers9,
        AN.H_9B,
        rng,
        rows_key="rows",
        with_leak=True,
    )
    _write_embeddings(root9, sorted(bank9["contexts"]), rng, "0.11.0")
    _write_cell_stores(
        root7,
        "analysis_tensors/vc2564/vc2564_bank.pt",
        "analysis_tensors/va2564",
        bank7,
        AN.LAYERS_7B,
        AN.H_7B,
        rng,
        rows_key="index",
        with_leak=False,
    )
    _write_embeddings(root7, sorted(bank7["contexts"]), rng, None)  # reference-by-pin

    ridge_dir = root / "fits"
    ridge_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.save(
        {
            "kind": "ridge",
            "W": torch.randn(AN.H_9B, AN.H_9B) * 1e-3,
            "xmu": torch.zeros(AN.H_9B),
            "xsd": torch.ones(AN.H_9B),
            "ymu": torch.zeros(AN.H_9B),
        },
        ridge_dir / f"L{LSTAR}.pt",
    )
    te_ids = [f"te{i:04d}" for i in range(50)]
    torch.save(_mk_preds(rng, te_ids, LSTAR, 0.8), ridge_dir / f"L{LSTAR}_preds.pt")
    preds7b = root / "preds7b"
    preds7b.mkdir(parents=True, exist_ok=True)
    n7 = len(bank7["contexts"])
    torch.save(
        {
            "context_ids": sorted(bank7["contexts"]),
            "tensor": rng.standard_normal((n7, AN.H_7B)).astype(np.float32),
        },
        preds7b / f"mapped_vc2564_{AN.ARM_7B_MATCHED}_L{AN.L19}.pt",
    )
    torch.save(
        _mk_preds(rng, te_ids, AN.L19, 0.6),
        preds7b / f"test_preds_{AN.ARM_7B_MATCHED}_L{AN.L19}.pt",
    )

    return [
        "--smoke",
        "--in-root-9b",
        str(root9),
        "--in-root-7b",
        str(root7),
        "--bank-9b",
        str(root / "bank9.json"),
        "--bank-7b",
        str(root / "bank7.json"),
        "--manip-9b",
        str(root / "manip9.json"),
        "--manip-7b",
        str(root / "manip7.json"),
        "--sweep-json",
        str(root / "sweep.json"),
        "--ridge-9b",
        str(ridge_dir / f"L{LSTAR}.pt"),
        "--preds-9b",
        str(ridge_dir / f"L{LSTAR}_preds.pt"),
        "--preds7b-dir",
        str(preds7b),
        "--ref7b-parent",
        str(root / "ref7b.json"),
        "--ref7b-parent-commit",
        REF7B_COMMIT,
        "--out-dir",
        str(root / "out"),
        "--stage-dir",
        str(root / "stage"),
        "--n-splits",
        "4",
    ]


def run_analysis(out_root: Path) -> None:
    """Analysis leg: real subprocess of the production entrypoint (rc asserted)."""
    root = out_root / "analysis"
    argv_tail = build_analysis_world(root)
    log_path = root / "analysis_smoke.log"
    cmd = [sys.executable, str(SCRIPTS / "issue2587_analysis.py"), *argv_tail]
    print(f"[smoke2587] analysis leg: {' '.join(cmd[:3])} ... ({len(cmd)} argv items)", flush=True)
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env={**os.environ}, capture_output=True, text=True, check=False
    )
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-40:])
        raise RuntimeError(f"analysis leg rc={proc.returncode}; log tail:\n{tail}")

    combined = proc.stdout + proc.stderr
    missing_sites = [s for s in ANALYSIS_SKIP_SITES if f"[smoke-blind-spot] {s}:" not in combined]
    assert not missing_sites, (
        f"registered skip sites did NOT fire in the child log: {missing_sites}"
    )
    assert "[smoke-blind-spot] bootstrap_null_b" in combined, "param-narrow warning missing"

    out = root / "out"
    doc = json.loads((out / "minpair_delta_2587.json").read_text())
    assert doc["meta"]["smoke"] is True
    bs = {e["site"]: e["kind"] for e in doc["meta"]["smoke_blind_spots"]}
    assert set(ANALYSIS_SKIP_SITES) <= set(bs), sorted(bs)
    assert bs["bootstrap_null_b"] == "param-narrowed", bs
    cm = json.loads((out / "crossmodel_contrasts.json").read_text())
    assert cm["meta"]["smoke"] is True and "h2" in cm
    n_rows = sum(1 for ln in (out / "perpair_2587.jsonl").open() if ln.strip())
    assert n_rows > 0, "perpair_2587.jsonl empty"
    ckpts = sorted(p.name for p in (out / "checkpoints").glob("*.json"))
    npzs = sorted(p.name for p in (out / "crossmodel_perdraw").glob("*.npz"))
    assert {"battery_qwen35_9b.json", "battery_qwen25_7b.json", "h1.json"} <= set(ckpts), ckpts
    assert npzs, "no perdraw npz artifacts"
    print(
        f"[smoke2587] analysis leg PASS rc=0: minpair_delta_2587.json + "
        f"crossmodel_contrasts.json + perpair ({n_rows} rows) + {len(ckpts)} ckpts + "
        f"{len(npzs)} npz under {out}; log={log_path}",
        flush=True,
    )


# ── judge leg (boundary fake installed in the CHILD, judge-child mode) ──


def build_judge_anchors(root: Path) -> Path:
    """Anchors JSONLs from the REAL pinned bank on the production smoke slice."""
    import issue2587_judge as J

    bk = J.B25._bk()
    values = bk.load_values()
    judged_axes = tuple(a for a in J.parent_judged_axes(bk) if a in J.SMOKE_CELLS)
    assert judged_axes, "SMOKE_CELLS covers no judged parent axis"
    cells_specs: dict[str, list[dict]] = {}
    for axis in judged_axes:
        cells_specs[axis] = J.judged_specs(bk, values, J.SMOKE_CARRIERS, (axis,))
    cells_specs[J.LANG_AXIS] = J.lang_specs(bk, J.SMOKE_CARRIERS)
    anchors_dir = root / "anchors"
    anchors_dir.mkdir(parents=True, exist_ok=True)
    for cell, specs in cells_specs.items():
        rows, seen = [], set()
        for s in specs:
            key = (s["context_id"], s["draw"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                json.dumps(
                    {
                        "context_id": s["context_id"],
                        "draw": s["draw"],
                        "text": f"A brief, on-topic reply for anchor {len(rows)}. Surely fine.",
                    }
                )
            )
        (anchors_dir / f"anchors_{cell}.jsonl").write_text("\n".join(rows) + "\n")
    return anchors_dir


def run_judge(out_root: Path) -> None:
    """Judge leg: child process runs the production entrypoint with the ONLY
    fake at the Batch-API boundary; parent asserts rc + artifact + pins."""
    root = out_root / "judge"
    root.mkdir(parents=True, exist_ok=True)
    anchors_dir = build_judge_anchors(root)
    out = root / "manipulation_check_2587.json"
    work = root / "work"
    record = root / "boundary_calls.jsonl"
    log_path = root / "judge_smoke.log"
    cmd = [
        sys.executable,
        str(SCRIPTS / "issue2587_smoke_run.py"),
        "judge-child",
        "--anchors-dir",
        str(anchors_dir),
        "--out",
        str(out),
        "--work-root",
        str(work),
        "--record",
        str(record),
    ]
    print(f"[smoke2587] judge leg: {' '.join(cmd[:3])} ...", flush=True)
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env={**os.environ}, capture_output=True, text=True, check=False
    )
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-40:])
        raise RuntimeError(f"judge leg rc={proc.returncode}; log tail:\n{tail}")

    doc = json.loads(out.read_text())
    meta = doc["meta"]
    assert meta["smoke"] is True and meta["dry_run"] is False
    assert meta["call_arithmetic"]["verified"] is False
    bs = {e["site"]: e["kind"] for e in meta["smoke_blind_spots"]}
    assert bs == {
        "call_arithmetic_1464": "assert-skipped",
        "smoke_slice_narrowing": "param-narrowed",
    }, bs
    assert doc["value_rows"] and doc["axis_rows"], "fire tables empty"
    calls = [json.loads(ln) for ln in record.open() if ln.strip()]
    assert len(calls) == 2, f"expected 2 boundary calls (both rubric families), got {len(calls)}"
    for c in calls:
        assert c["judge_model"] == "claude-sonnet-4-5-20250929", c
        assert c["max_tokens"] == 1024 and c["threshold_base"] == 0, c
    scores_path = work / "raw" / "judge_scores.jsonl"
    n_scored = sum(1 for ln in scores_path.open() if ln.strip())
    assert n_scored > 0, "judge_scores.jsonl empty"
    raw = json.loads((work / "raw" / "judge_raw_manipcheck.json").read_text())
    assert raw["all_scores"], "boundary save_raw empty"
    print(
        f"[smoke2587] judge leg PASS rc=0: sentinel={out} ({len(doc['value_rows'])} value rows), "
        f"{n_scored} scored rows, 2 boundary calls pinned "
        f"(model/max_tokens/threshold_base); log={log_path}",
        flush=True,
    )


def _judge_child(args: argparse.Namespace) -> None:
    """CHILD-process mode: install the boundary fake, then run the judge
    script as __main__ so its SystemExit(0) is THIS process's exit code."""
    from unittest import mock

    from explore_persona_space.eval import batch_judge as BJ

    record_path = Path(args.record)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    real = BJ.judge_completions_batch

    def fake(**kw):
        # Pin the plan invariants AT the true network boundary.
        assert kw["judge_model"] == "claude-sonnet-4-5-20250929", kw["judge_model"]
        assert kw["max_tokens"] == 1024, kw["max_tokens"]
        assert kw["threshold_base"] == 0, kw["threshold_base"]
        all_scores: dict[str, dict] = {}
        for item_id, qmap in kw["completions"].items():
            for _q, comps in qmap.items():
                for ci in range(len(comps)):
                    all_scores[f"{item_id}__00000__{ci:02d}"] = {
                        "score": 88,
                        "stop_reason": "end_turn",
                    }
        save_raw = Path(kw["save_raw"])
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(
            json.dumps(
                {
                    "all_scores": all_scores,
                    "routing": {"note": "faked at the Batch-API boundary (judge-child)"},
                }
            )
        )
        with record_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "n_items": len(kw["completions"]),
                        "judge_model": kw["judge_model"],
                        "max_tokens": kw["max_tokens"],
                        "threshold_base": kw["threshold_base"],
                        "save_raw": str(save_raw),
                    }
                )
                + "\n"
            )
        return {k: 88.0 for k in all_scores}

    BJ.judge_completions_batch = mock.create_autospec(real, side_effect=fake)
    sys.argv = [
        "issue2587_judge.py",
        "--smoke",
        "--skip-upload",
        "--anchors-dir",
        args.anchors_dir,
        "--out",
        args.out,
        "--work-root",
        args.work_root,
    ]
    runpy.run_path(str(SCRIPTS / "issue2587_judge.py"), run_name="__main__")


# ── CLI ─────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("leg", choices=("analysis", "judge", "all", "judge-child"))
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--anchors-dir", default=None, help="judge-child: anchors dir")
    ap.add_argument("--out", default=None, help="judge-child: sentinel out path")
    ap.add_argument("--work-root", default=None, help="judge-child: judge work root")
    ap.add_argument("--record", default=None, help="judge-child: boundary-call record JSONL")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0
    if args.leg == "judge-child":
        for req in ("anchors_dir", "out", "work_root", "record"):
            assert getattr(args, req), f"judge-child requires --{req.replace('_', '-')}"
        _judge_child(args)  # runpy propagates the judge's SystemExit
        return 0
    out_root = Path(args.out_root)
    if args.leg in ("analysis", "all"):
        run_analysis(out_root)
    if args.leg in ("judge", "all"):
        run_judge(out_root)
    print(f"[smoke2587] done leg={args.leg} out_root={out_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
