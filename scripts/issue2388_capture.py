#!/usr/bin/env python
"""#2388 P2: teacher-forced capture + TF secondary-DV margins for the new surfaces.

Per-benchmark phases over the P1 generation outputs (``scripts/issue2388_gen.py``
rollouts JSONLs), reusing the #1739 capture core
(``experiments/issue_1739/capture.py``) — same store layout, BPE-seam
discipline, resume, and completeness reconciliation:

  capture    build (payload, meta) rows from items x K rollouts and capture
             kinds ``context_end`` / ``t1`` / ``t_last`` into
             ``<store-root>/<benchmark>/`` (plan section 4 pooling fork: the
             per-rollout store deliberately OMITS ``prefix_end`` — storing it
             per rollout would push the store past the ~130 GB MooseFS quota,
             plan section 9; ``t_last`` is the NEW last-answer-token pooling
             companion).
  tf-margin  teacher-forced secondary DV per item (plan section 6):
             math -> ln P(gold boxed answer); MCQ -> ln P(gold option) minus
             logsumexp over distractor options; code -> ln P(canonical
             reference solution), FLAGGED ROUGH (lcb_v5 has no reference
             except the LeetCode dedup overlap).
  upload     tar each benchmark store dir + upload to the HF data repo, plus
             the tf_margin JSONs; exact-set post-upload verify.

Runs POD-SIDE (GPU: bf16 HF forwards). ``--import-check`` is the CPU
pre-flight (argcheck + deferred-import execution). CONTENT HYGIENE: logs
carry ids/counts only, never row text.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """#823: script mode puts scripts/ on sys.path[0], not the repo root."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import issue2388_gen as G  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.experiments.issue_1739.capture import (  # noqa: E402
    _token_ids,
    capture_row_ids_and_positions,
    capture_rows_to_store,
    load_capture_model,
    teacher_forced_ln_logp,
)
from explore_persona_space.experiments.issue_1739.constants import MODEL_NAME  # noqa: E402
from explore_persona_space.experiments.issue_1739.generation import (  # noqa: E402
    INSTRUCT_REVISION,
    MAX_MODEL_LEN,
    get_tokenizer,
)

# Store kinds (plan section 4): v_C at context_end + BOTH answer poolings.
CAPTURE_KINDS = ("context_end", "t1", "t_last")
# Plan section 6.5/10 artifact contract: analysis tensors live under
# issue2388_correctness/analysis_tensors/ (r1 Codex artifact-path blocker).
HF_STORE_PREFIX = "issue2388_correctness/analysis_tensors/capture_store"
HF_DV_PREFIX = "issue2388_correctness/dv"
DEFAULT_STORE_ROOT = "/workspace/store_2388"
DEFAULT_DV_ROOT = "eval_results/issue_2388/dv"
# Per-benchmark store floor: rows x kinds x 28 x 3584 x 2B + sidecars. The
# largest benchmark (math_full, 62.5k rows) needs ~38 GB; assert generously.
STORE_HEADROOM_GB = {"math_full": 45.0, "mmlu_pro_full": 40.0}
STORE_HEADROOM_DEFAULT_GB = 15.0
TF_CHUNK = 200  # per-unit persistence grain (code-style checkpoint rule, T2)


# ---------------------------------------------------------------------------
# capture rows
# ---------------------------------------------------------------------------


def build_capture_rows(
    items: list[dict],
    rolls: dict[str, list[str]],
    tokenizer,
    *,
    benchmark: str,
    max_model_len: int = MAX_MODEL_LEN,
) -> tuple[list[tuple[dict, dict]], int, list[dict]]:
    """(payload, meta) rows for items x K rollouts, chat-templated as gen was.

    The prompt segment is rebuilt through the SAME ``apply_chat_template`` call
    ``issue2388_gen.phase_gen`` fed to vLLM, so the teacher-forced prompt ids
    are bit-identical to what generation consumed (BPE-seam discipline).
    Over-budget rows are DROPPED with a digest record (id + token count only).
    """
    missing = [it["item_id"] for it in items if it["item_id"] not in rolls]
    if missing:
        raise RuntimeError(
            f"{benchmark}: {len(missing)} items lack rollouts (e.g. {missing[:3]}) — "
            "run issue2388_gen --phase gen first"
        )
    rows: list[tuple[dict, dict]] = []
    over_budget: list[dict] = []
    surface = G.surface_of(benchmark)
    for it in items:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        completions = rolls[it["item_id"]]
        if len(completions) != G.K_ROLLOUTS:
            raise RuntimeError(
                f"{benchmark}/{it['item_id']}: {len(completions)} rollouts != {G.K_ROLLOUTS}"
            )
        for k, comp in enumerate(completions):
            payload = {"prefix_text": "", "prompt_text": prompt, "completion": comp}
            meta = {
                "context_id": it["item_id"],
                "benchmark": benchmark,
                "surface": surface,
                "rollout_k": k,
                "is_eval_only": False,
                "source_file": f"{benchmark}.jsonl",
            }
            try:
                _, pos = capture_row_ids_and_positions(
                    tokenizer, "", prompt, comp, max_model_len=max_model_len
                )
            except ValueError as e:
                # Only the BUDGET ValueErrors are drop-class ("exceeding
                # max_model_len" / "exceeding prompt budget"); any other
                # ValueError is a tokenizer-side bug and must crash (r1 g5).
                if "exceeding" not in str(e):
                    raise
                over_budget.append({"item_id": it["item_id"], "rollout_k": k})
                continue
            rows.append((payload, dict(meta, n_row_tokens=pos["n_total"])))
    if not rows:
        raise RuntimeError(f"{benchmark}: 0 in-budget capture rows")
    return rows, len(over_budget), over_budget


def _fingerprint(benchmark: str, roll_path: Path | None = None) -> str:
    """Machine-stable resume fingerprint: generating parameters + the ROLLOUT
    file's content digest (bytes read from disk — hash-safe per the float-key
    rule). Re-generated rollouts (--regen-cap-hit) then refuse a stale-store
    resume instead of silently mixing captures (r2 long-loop-restartability).
    """
    fp = (
        f"i2388|{benchmark}|k={G.K_ROLLOUTS}|kinds={','.join(CAPTURE_KINDS)}"
        f"|rev={INSTRUCT_REVISION[:12]}"
    )
    if roll_path is not None and roll_path.exists():
        import hashlib

        h = hashlib.sha256()
        with roll_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        fp += f"|rolls={h.hexdigest()[:12]}"
    return fp


def phase_capture(args) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    benchmark = args.benchmark
    out_root = Path(args.out_root)
    store_dir = Path(args.store_root) / benchmark
    need_gb = STORE_HEADROOM_GB.get(benchmark, STORE_HEADROOM_DEFAULT_GB)
    assert_out_root_headroom(Path(args.store_root), need_gb, phase=f"capture-{benchmark}")

    items = G.LOADERS[benchmark]()
    if benchmark == "lcb_v5":
        items = G._apply_dedup(items, out_root)
    tokenizer = get_tokenizer()
    rolls = G._load_done_rollouts(G._rollouts_path(out_root, benchmark))
    if args.smoke:
        items = items[: G.SMOKE_N]
    rows, n_over, over_digest = build_capture_rows(
        rolls=rolls, items=items, tokenizer=tokenizer, benchmark=benchmark
    )
    print(
        f"[capture] {benchmark}: {len(rows)} rows ({len(items)} items x {G.K_ROLLOUTS}), "
        f"{n_over} over budget",
        flush=True,
    )
    if n_over:
        digest_path = store_dir / "_over_budget_rows.json"
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        digest_path.write_text(json.dumps(over_digest, indent=1))

    model = load_capture_model(device=args.device)
    manifest = capture_rows_to_store(
        rows,
        store_dir=store_dir,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batch_size=args.batch_size,
        fingerprint=_fingerprint(benchmark, G._rollouts_path(out_root, benchmark)),
        kinds=CAPTURE_KINDS,
        n_over_budget=n_over,
        n_rollout_files=1,
    )
    print(
        f"[capture] {benchmark} complete: {manifest['realized_total_rows']} realized rows "
        f"in {manifest['n_shards']} shards -> {store_dir}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# tf-margin (secondary DV; plan section 6)
# ---------------------------------------------------------------------------


def _tf_out_dir(dv_root: Path, benchmark: str) -> Path:
    """TF-margin rows live under the DV root (plan section 6.5: ``dv/*/tf_margin``),
    NOT the gen root (r1 Codex artifact-path blocker)."""
    return dv_root / G.surface_of(benchmark) / "tf_margin"


def _chat_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )


def _code_canonicals(benchmark: str) -> dict[str, list[tuple[str, str]]]:
    """Canonical reference solutions per item (the code-control canon builders)."""
    import issue2388_code_control as CC

    key = {
        "humaneval": "humaneval",
        "mbpp_full": "mbpp",
        "bigcodebench_full": "bigcodebench",
        "lcb_v5": "lcb_v5",
        "leetcode": "leetcode",
        # fork-5 contingency benchmark (r2 bug-class sweep: an activated APPS
        # tf-margin leg must not KeyError on the canonical map).
        "apps_intro": "apps_intro",
    }[benchmark]
    return CC.BENCHES[key]["canon"]()


def _tf_units(benchmark: str, items: list[dict], tokenizer) -> list[dict]:
    """One scoring unit per item: {'item_id', 'pairs': [(label, prompt, completion)]}.

    math: single gold completion. MCQ: one completion per option letter (the
    margin is computed at aggregation). code: one completion per canonical
    composition (ROUGH — the reference is not the model's own distribution).
    """
    units: list[dict] = []
    if benchmark == "math_full":
        for it in items:
            prompt = _chat_prompt(tokenizer, it["prompt"])
            comp = f"The final answer is \\boxed{{{it['gold']}}}."
            units.append({"item_id": it["item_id"], "pairs": [("gold", prompt, comp)]})
    elif benchmark == "mmlu_pro_full":
        for it in items:
            prompt = _chat_prompt(tokenizer, it["prompt"])
            letters = [chr(ord("A") + i) for i in range(int(it["n_options"]))]
            pairs = [(letter, prompt, f"Answer: {letter}") for letter in letters]
            units.append({"item_id": it["item_id"], "gold": it["gold"], "pairs": pairs})
    else:  # code benchmarks
        canon = _code_canonicals(benchmark)
        for it in items:
            cands = canon.get(it["item_id"])
            if not cands:
                units.append({"item_id": it["item_id"], "pairs": []})  # no reference (lcb)
                continue
            prompt = _chat_prompt(tokenizer, it["prompt"])
            pairs = [
                (label, prompt, f"```python\n{sol}\n```")
                for label, sol in cands
                if sol and sol.strip()
            ]
            units.append({"item_id": it["item_id"], "pairs": pairs})
    return units


def _write_surface_tf_aggregate(dv_root: Path, surface: str) -> Path:
    """Plan-contract aggregate ``dv/<surface>/tf_margin.json`` from whatever
    per-benchmark JSONLs are complete (each has a ``*_tf_summary.json``)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    tf_dir = dv_root / surface / "tf_margin"
    rows: list[dict] = []
    included: dict[str, dict] = {}
    for summ_p in sorted(tf_dir.glob("*_tf_summary.json")):
        summ = json.loads(summ_p.read_text())
        bench = summ["benchmark"]
        jsonl = tf_dir / f"{bench}_tf.jsonl"
        with jsonl.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
        included[bench] = {"n_rows": summ["n_rows"], "recipe": summ["recipe"]}
    if not rows:
        raise RuntimeError(f"no completed tf-margin benchmarks under {tf_dir}")
    payload = {
        "surface": surface,
        "benchmarks_included": included,
        "n_rows": len(rows),
        "rows": rows,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.update(as_metadata_dict(git_provenance(), phase=f"tf-margin-agg-{surface}"))
    out = dv_root / surface / "tf_margin.json"
    with atomic_replace(out) as tmp:
        tmp.write_text(json.dumps(payload))
    print(f"[tf-margin] aggregate: {len(rows)} rows ({sorted(included)}) -> {out}", flush=True)
    return out


def phase_tf_margin(args) -> None:
    benchmark = args.benchmark
    out_root = Path(args.out_root)
    dv_root = Path(args.dv_root)
    items = G.LOADERS[benchmark]()
    if benchmark == "lcb_v5":
        items = G._apply_dedup(items, out_root)
    if args.smoke:
        items = items[: G.SMOKE_N]
    tokenizer = get_tokenizer()
    units = _tf_units(benchmark, items, tokenizer)

    tf_dir = _tf_out_dir(dv_root, benchmark)
    tf_dir.mkdir(parents=True, exist_ok=True)
    rows_path = tf_dir / f"{benchmark}_tf.jsonl"
    done: set[str] = set()
    if rows_path.exists():
        with rows_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["item_id"])
    pending = [u for u in units if u["item_id"] not in done]
    print(f"[tf-margin] {benchmark}: {len(pending)}/{len(units)} units pending", flush=True)

    model = load_capture_model(device=args.device) if pending else None
    t0 = time.time()
    for start in range(0, len(pending), TF_CHUNK):
        chunk = pending[start : start + TF_CHUNK]
        flat = [(u["item_id"], label, p, c) for u in chunk for (label, p, c) in u["pairs"]]
        lps = teacher_forced_ln_logp(
            [(p, c) for (_, _, p, c) in flat],
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            batch_size=args.batch_size,
        )
        # teacher_forced_ln_logp returns the per-token MEAN; the documented
        # statistics are TOTAL ln-probs, so un-normalize by the SAME segment
        # token count the scorer divided by (r1 g5: the mean-based MCQ margin
        # was a temperature-L-scaled variant of the documented read).
        by_item: dict[str, dict[str, float]] = {}
        by_item_mean: dict[str, dict[str, float]] = {}
        for (item_id, label, _, comp), lp in zip(flat, lps, strict=True):
            n_tok = len(_token_ids(tokenizer, comp))
            by_item.setdefault(item_id, {})[label] = lp * n_tok
            by_item_mean.setdefault(item_id, {})[label] = lp
        with rows_path.open("a", encoding="utf-8") as fh:
            for u in chunk:
                lp_map = by_item.get(u["item_id"], {})
                row: dict = {
                    "item_id": u["item_id"],
                    "benchmark": benchmark,
                    "lp": lp_map,  # per-label TOTAL ln P(completion | prompt)
                    "lp_per_token": by_item_mean.get(u["item_id"], {}),
                }
                if benchmark == "mmlu_pro_full":
                    gold = u["gold"]
                    others = [v for k, v in lp_map.items() if k != gold]
                    if gold not in lp_map or not others:
                        # Fires on an EMPTY lp_map too (r1 g5: the old
                        # `and lp_map` guard made this raise unreachable).
                        raise RuntimeError(f"tf-margin: malformed option set for {u['item_id']}")
                    lse = max(others) + math.log(sum(math.exp(v - max(others)) for v in others))
                    row["tf_margin"] = lp_map[gold] - lse
                elif benchmark == "math_full":
                    row["tf_gold_ln_logp"] = lp_map["gold"]
                    row["tf_gold_ln_logp_per_token"] = by_item_mean[u["item_id"]]["gold"]
                else:
                    # Reference solutions vary hugely in length, so the code
                    # read stays PER-TOKEN (comparable across compositions) —
                    # ROUGH either way (not the model's own distribution).
                    mean_map = by_item_mean.get(u["item_id"], {})
                    row["tf_reference_ln_logp"] = max(mean_map.values()) if mean_map else None
                    row["tf_reference_rough"] = True
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[tf-margin] unit {min(start + TF_CHUNK, len(pending))}/{len(pending)} "
            f"{benchmark} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    n_rows = sum(1 for line in rows_path.open(encoding="utf-8") if line.strip())
    if n_rows < len(units):
        raise RuntimeError(f"tf-margin {benchmark}: {n_rows} rows < {len(units)} units")
    summary = {
        "benchmark": benchmark,
        "n_units": len(units),
        "n_rows": n_rows,
        "recipe": {
            "math_full": "total ln P(boxed gold sentence | chat prompt); per-token mean stored",
            "mmlu_pro_full": ("total ln P('Answer: <gold>') - logsumexp over distractor totals"),
            "code": (
                "max over canonical compositions of PER-TOKEN mean ln P(fenced reference) "
                "— ROUGH (length-normalized; references are not the model's own distribution)"
            ),
        }[benchmark if benchmark in ("math_full", "mmlu_pro_full") else "code"],
        "model": MODEL_NAME,
        "revision": INSTRUCT_REVISION,
    }
    summary.update(as_metadata_dict(git_provenance(), phase=f"tf-margin-{benchmark}"))
    (tf_dir / f"{benchmark}_tf_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[tf-margin] {benchmark} complete: {n_rows} rows -> {rows_path}", flush=True)
    _write_surface_tf_aggregate(dv_root, G.surface_of(benchmark))


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def _upload_state_descriptor(
    store_root: Path, dv_root: Path, benchmarks: list[str], hf_root: str
) -> dict:
    """Local-file state the upload phase is a pure function of (r4 Minor
    capture-upload-phase-not-skippable): per-benchmark capture manifest + tar,
    tf_margin dir files, per-surface aggregates — each as (size, mtime_ns).
    A sentinel written after a VERIFIED upload that exact-matches this
    descriptor at re-entry means a rerun would re-transfer identical bytes,
    so the phase may skip (``--force-upload`` overrides)."""

    def _stat(p: Path) -> list[int]:
        st = p.stat()
        return [int(st.st_size), int(st.st_mtime_ns)]

    state: dict = {"hf_root": hf_root, "benchmarks": sorted(benchmarks), "files": {}}
    for benchmark in benchmarks:
        surface = G.surface_of(benchmark)
        for p in (
            store_root / benchmark / "_capture_manifest.json",
            store_root / f"{benchmark}.tar",
        ):
            if p.exists():
                state["files"][str(p)] = _stat(p)
        tf_dir = _tf_out_dir(dv_root, benchmark)
        if tf_dir.exists():
            for p in sorted(tf_dir.iterdir()):
                state["files"][str(p)] = _stat(p)
        agg = dv_root / surface / "tf_margin.json"
        if agg.exists():
            state["files"][str(agg)] = _stat(agg)
    return state


def phase_upload(args) -> None:
    """Tar + upload each benchmark's store; folder-upload the tf_margin dirs.

    Smoke runs land under a ``_smoke``-suffixed HF ROOT (same rule as gen's
    phase_upload: HF prefixes are production artifacts). A tar older than the
    store's manifest is REBUILT — a presence-only tar resume would upload
    pre-repair bytes (r1 g5 Minor 2). A rerun whose local inputs are byte-
    stat-identical to the last VERIFIED upload SKIPS the multi-GB Hub
    transfers via the ``_upload_state*.json`` sentinel; ``--force-upload``
    re-uploads regardless (r4 Minor capture-upload-phase-not-skippable).
    """
    import tarfile

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    dv_root = Path(args.dv_root)
    store_root = Path(args.store_root)
    hf_root = "issue2388_correctness" + ("_smoke" if args.smoke else "")
    store_prefix = HF_STORE_PREFIX.replace("issue2388_correctness", hf_root, 1)
    dv_prefix = HF_DV_PREFIX.replace("issue2388_correctness", hf_root, 1)
    if args.benchmark:
        benchmarks = [args.benchmark]
    elif args.surface == "code":
        # Fork-5 contingency roster (r3 Critical 3): the exact upload set comes
        # from the BINDING gate verdict — never the static surface roster (a
        # DROPPED BCB has no store, so requiring its manifest deadlocks the
        # DROP->APPS branch) and never file existence (a stale APPS manifest
        # must not ride a KEEP-branch upload).
        benchmarks = G.code_roster_from_gate_fields(G.load_gate(Path(args.out_root)))
    else:
        benchmarks = sorted(G.SURFACES[args.surface])
    sentinel_p = store_root / ("_upload_state" + ("_smoke" if args.smoke else "") + ".json")
    if sentinel_p.exists() and not args.force_upload:
        prior_state = json.loads(sentinel_p.read_text())
        if prior_state == _upload_state_descriptor(store_root, dv_root, benchmarks, hf_root):
            print(
                f"[upload] {sentinel_p.name} matches local state — verified upload already "
                "landed; skip (re-run with --force-upload to re-transfer)",
                flush=True,
            )
            return
        print("[upload] upload sentinel present but local state changed — re-uploading", flush=True)
    expected: list[str] = []
    for benchmark in benchmarks:
        surface = G.surface_of(benchmark)
        store_dir = store_root / benchmark
        manifest_p = store_dir / "_capture_manifest.json"
        if not manifest_p.exists():
            raise RuntimeError(f"{store_dir} has no _capture_manifest.json — capture incomplete")
        tar_path = store_root / f"{benchmark}.tar"
        if not tar_path.exists() or tar_path.stat().st_mtime < manifest_p.stat().st_mtime:
            with atomic_replace(tar_path) as tmp:
                with tarfile.open(tmp, "w") as tf:
                    tf.add(store_dir, arcname=benchmark)
        dest = f"{store_prefix}/{surface}/{benchmark}.tar"
        # UPLOAD_LOOP_EXEMPT: bounded — at most 6 multi-GB benchmark tars, never a per-file storm
        out = hub._upload(
            tar_path,
            hub.DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=dest,
            upload_as_file=True,
        )
        if not out:
            raise RuntimeError(f"store tar upload returned empty path for {benchmark}")
        expected.append(dest)
        # The tar is a pure upload vehicle: retaining all of them alongside the
        # stores doubles the phase's footprint past the 200G volume (R8 ENOSPC
        # at leetcode). A re-entry rebuilds any needed tar from the store via
        # the mtime rule above.
        tar_path.unlink()
        print(f"[upload] {benchmark}: local tar removed after upload", flush=True)
        tf_dir = _tf_out_dir(dv_root, benchmark)
        if tf_dir.exists() and any(tf_dir.iterdir()):
            out = hub._upload(
                tf_dir,
                hub.DEFAULT_DATASET_REPO,
                repo_type="dataset",
                path_in_repo=f"{dv_prefix}/{surface}/tf_margin",
            )
            if not out:
                raise RuntimeError(f"tf_margin upload returned empty path for {benchmark}")
            expected.extend(f"{dv_prefix}/{surface}/tf_margin/{p.name}" for p in tf_dir.iterdir())
        agg = dv_root / surface / "tf_margin.json"
        if agg.exists():
            dest_agg = f"{dv_prefix}/{surface}/tf_margin.json"
            # UPLOAD_LOOP_EXEMPT: bounded — one aggregate JSON per surface (<=4), never a storm
            out = hub._upload(
                agg,
                hub.DEFAULT_DATASET_REPO,
                repo_type="dataset",
                path_in_repo=dest_agg,
                upload_as_file=True,
            )
            if not out:
                raise RuntimeError(f"tf_margin aggregate upload returned empty path for {surface}")
            expected.append(dest_agg)
        print(f"[upload] {benchmark}: store tar + tf_margin uploaded", flush=True)

    missing = hub.verify_repo_paths_uploaded(
        HfApi(),
        hub.DEFAULT_DATASET_REPO,
        sorted(set(expected)),
        path_in_repo=hf_root,
        repo_type="dataset",
    )
    if missing:
        raise RuntimeError(f"post-upload verify: {len(missing)} paths missing: {missing[:5]}")
    # Sentinel AFTER the exact-set verify, recomputed post-tar-rebuild so the
    # recorded stats are the uploaded tars' — the skip contract above.
    sentinel_p.write_text(
        json.dumps(_upload_state_descriptor(store_root, dv_root, benchmarks, hf_root), indent=1)
    )
    print(f"[upload] verified {len(set(expected))} paths on the Hub", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

PHASES = {"capture": phase_capture, "tf-margin": phase_tf_margin, "upload": phase_upload}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace("%", "%%"))
    ap.add_argument("--phase", choices=sorted(PHASES), help="phase to run")
    ap.add_argument("--benchmark", choices=sorted(G.LOADERS), default=None)
    ap.add_argument("--surface", choices=sorted(G.SURFACES), default=None)
    ap.add_argument("--out-root", default=str(G.OUT_ROOT), help="gen out-root (P1 outputs)")
    ap.add_argument("--dv-root", default=DEFAULT_DV_ROOT, help="DV root (tf_margin destination)")
    ap.add_argument("--store-root", default=DEFAULT_STORE_ROOT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="cap items to SMOKE_N + smoke roots")
    ap.add_argument(
        "--force-upload",
        action="store_true",
        help="re-run Hub transfers even when the verified-upload sentinel matches local state",
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    args = ap.parse_args(argv)

    if args.list_phases:
        print(" ".join(sorted(PHASES)))
        return 0
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute the deferred imports (except the GPU-fenced model load).
        import tarfile  # noqa: F401

        import issue2388_code_control  # noqa: F401
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.orchestrate import hub
        from explore_persona_space.orchestrate.preflight import (  # noqa: F401
            assert_out_root_headroom,
        )
        from explore_persona_space.orchestrate.provenance import (  # noqa: F401
            as_metadata_dict,
            git_provenance,
        )

        assert callable(hub._upload) and hub.DEFAULT_DATASET_REPO
        print("[import-check] ok (GPU-fenced: load_capture_model — model load not executed)")
        return 0

    if not args.phase:
        raise SystemExit("--phase is required (or --import-check / --list-phases)")
    if args.phase in ("capture", "tf-margin") and not args.benchmark:
        raise SystemExit(f"--phase {args.phase} requires --benchmark")
    if args.phase == "upload" and not (args.benchmark or args.surface):
        raise SystemExit("--phase upload requires --benchmark or --surface")
    if args.smoke:
        # Smoke roots: never overwrite production paths (same convention as gen;
        # phase_upload additionally suffixes the HF root).
        if args.out_root == str(G.OUT_ROOT):
            args.out_root = str(Path("eval_results/issue_2388/gen_smoke"))
        if args.store_root == DEFAULT_STORE_ROOT:
            args.store_root = DEFAULT_STORE_ROOT + "_smoke"
        if args.dv_root == DEFAULT_DV_ROOT:
            args.dv_root = DEFAULT_DV_ROOT + "_smoke"
    PHASES[args.phase](args)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    # Explicit exit after flush: heavy C-extension teardown (torch/datasets)
    # can rewrite the rc at finalization (#1689 phased-dispatcher rule).
    sys.exit(main())
