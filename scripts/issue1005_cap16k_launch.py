#!/usr/bin/env python3
# Intentional Unicode (→) in docstrings + log messages.
"""Issue #1005 follow-up `cap16k-compliance-reread` launcher (amendment plan v4).

Instance-side single entrypoint for the ONE-variable re-run: the 97 residual
cap-hit rows (``finish_reason == "length"`` at the parent's 8,192 production
cap) are re-generated at 16,384 and the compliance statistic C + the frozen-
layer sensitivity fits are re-read with the recovered rows included. Phases:

1. **stage** — the 50 pinned rollout files + the 52 store files from HF
   revision ``621b370c`` into the driver's consumer layout (scoped
   ``list_repo_tree`` + per-file ``hf_hub_download`` — never
   ``snapshot_download`` on the ~1M-file data repo, gotcha #833). Smoke
   substitutes ``--stage-from-local`` (the Hub-boundary fake; the hub→local
   mapping helper is shared + unit-tested against the real hub layout).
2. **seed** — ``run_state.json`` seeded from a staged rollout blob's own
   recorded fields so the driver's gate section RESUMES (no gate slice is
   re-generated); every staged blob's ``model_revision`` is asserted.
3. **probe** — the (h)(iv) staging probe: ONE staged store blob opened via the
   driver's own ``reusable_store_blob`` validator BEFORE any GPU work.
4. **driver A (GPU)** — ``--phases extract --skip-gen --force-regen-16k
   --hf-stage-suffix _16k``: forced Phase P regen at 16,384 + regen accounting
   + C re-read + rollout upload to ``thinking_rollouts_16k`` + digest-triggered
   recapture of affected contexts + store upload to ``percq_summaries_16k``.
5. **driver B (GPU)** — ``--phases f1 --skip-gen --layers <frozen> --no-upload``:
   the inherited f1 machinery restricted to the parent's frozen layer indices
   (avg_q 18 / indiv 26, read from the committed ``bootstrap_deltaskill.json``).
6. **frozen-layer check** — the re-run boot blob's
   ``primary_frozen_direct_best_layer`` per regime vs the parent's; a mismatch
   is a reported DIAGNOSTIC (the pinned-index f4 explicit-layer reads carry the
   sensitivity headline, plan §2), never a crash.
7. **upload** — fit outputs + eval JSONs scoped to
   ``issue1005_cot_decomposition_r1/fit_results_16k/`` (one bulk commit +
   scoped verify), then the terminal results sentinel + ``[phase=done]``.

F4 (Δ_fam) runs VM-side afterwards (plan §3) — not here.

Usage::

    # production (GCP capture-7b lane, via dispatch_issue.py --workload-cmd):
    uv run python scripts/issue1005_cap16k_launch.py --gpu

    # CPU smoke (after one driver smoke run produced <parent>/ artifacts):
    uv run python scripts/issue1005_cap16k_launch.py --smoke \\
        --stage-from-local /tmp/issue-1005-smoke/parent_data \\
        --model /tmp/issue-1005-smoke/tiny_model --synthetic-completions \\
        --contexts 6 --probes 4 --layers 0 1 --n-perms 10 --n-boot 50 \\
        --skip-upload --out-dir /tmp/issue-1005-smoke/cap16k_data \\
        --eval-out /tmp/issue-1005-smoke/cap16k_eval \\
        --figures-dir /tmp/issue-1005-smoke/cap16k_figs \\
        --log-dir /tmp/issue-1005-smoke/logs
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", "/workspace/.cache/huggingface"))

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

from issue594_common import probes_hash  # noqa: E402
from issue928_common import (  # noqa: E402
    HF_DATA_REPO,
    context_order_and_families,
    dump_json,
    load_probe_pool,
    reproducibility_metadata,
    resolve_battery,
    upload_folder_scoped_verify,
    write_sentinel,
)
from issue928_extract_thinking_store import (  # noqa: E402
    reusable_store_blob,
    rollout_content_digest,
)
from issue1005_common import (  # noqa: E402
    HF_PREFIX_1005,
    MODEL_REVISION,
    RAW_COMPLETIONS_PREFIX_1005,
    STORE_PREFIX_1005,
    SUMMARY_NAMES_1005,
    THINKING_MODEL,
)

from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1005_cap16k")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PARENT_STORE_REVISION = "621b370c668d5a1df0c158aa522ef9d046c4b3c2"  # plan v4 §2 pinned inputs
FIT_RESULTS_16K_PREFIX = f"{HF_PREFIX_1005}/fit_results_16k"
REGIMES = ("avg_q", "indiv")


def phase(name: str) -> None:
    """Emit a poll_pipeline.py-parseable phase line."""
    print(f"[phase={name}]", flush=True)


def hub_to_local_relpath(hub_path: str) -> Path | None:
    """Map one pinned-input Hub path to the driver's consumer-layout relpath.

    The parent's store upload nested the folder under its own name (folder
    ``store/`` uploaded at ``path_in_repo=.../store/percq_summaries``), so the
    Hub layout is ``.../store/percq_summaries/{manifest.json,
    row_bookkeeping.json, percq_summaries/<ctx>.pt}`` — verified live at the
    pinned revision. Consumers open ``<out>/store/<rel>`` and
    ``<out>/raw_completions/thinking_rollouts/<name>.json``. Returns None for
    a path outside both prefixes (fail-loud at the caller).
    """
    roll_prefix = RAW_COMPLETIONS_PREFIX_1005 + "/"
    store_prefix = STORE_PREFIX_1005 + "/"
    if hub_path.startswith(roll_prefix):
        return Path("raw_completions") / "thinking_rollouts" / hub_path[len(roll_prefix) :]
    if hub_path.startswith(store_prefix):
        return Path("store") / hub_path[len(store_prefix) :]
    return None


def _stage_from_hub(out_dir: Path, revision: str) -> list[Path]:
    """Stage the pinned rollout + store files from the Hub into ``out_dir``.

    Scoped ``list_repo_tree`` per prefix at the pinned revision (never a bare
    full listing / ``snapshot_download`` — the ~1M-file data repo, #833), then
    per-file ``hf_hub_download`` (≤6 threads) wrapped in
    ``hub.retry_transient``, ``os.replace``d into the consumer layout.
    Idempotent: an existing non-empty destination is skipped (spot-preemption
    resume). Returns the staged destination paths.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    plans: list[tuple[str, Path]] = []
    for prefix in (RAW_COMPLETIONS_PREFIX_1005, STORE_PREFIX_1005):
        entries = hub.retry_transient(
            lambda p=prefix: list(
                api.list_repo_tree(
                    HF_DATA_REPO,
                    path_in_repo=p,
                    repo_type="dataset",
                    recursive=True,
                    revision=revision,
                )
            ),
            what=f"list_repo_tree {prefix}@{revision[:12]}",
        )
        for e in entries:
            if getattr(e, "blob_id", None) is None:
                continue  # folder entry
            rel = hub_to_local_relpath(e.path)
            if rel is None:
                raise RuntimeError(f"unmappable pinned-input Hub path: {e.path}")
            plans.append((e.path, out_dir / rel))
    n_roll = sum(1 for _hp, d in plans if d.suffix == ".json" and "thinking_rollouts" in str(d))
    n_store = len(plans) - n_roll
    if n_roll != 50 or n_store != 52:
        raise RuntimeError(
            f"pinned-input listing mismatch at {revision[:12]}: {n_roll} rollout files "
            f"(want 50), {n_store} store files (want 52) — refusing to stage"
        )

    stage_tmp = Path(tempfile.mkdtemp(prefix="cap16k_hfstage_", dir=str(out_dir)))

    def _fetch(hub_path: str, dest: Path) -> Path:
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        got = hub.retry_transient(
            lambda: hf_hub_download(
                HF_DATA_REPO,
                hub_path,
                repo_type="dataset",
                revision=revision,
                local_dir=str(stage_tmp),
            ),
            what=f"hf_hub_download {hub_path}",
        )
        os.replace(got, dest)
        return dest

    with ThreadPoolExecutor(max_workers=6) as ex:
        dests = list(ex.map(lambda t: _fetch(*t), plans))
    shutil.rmtree(stage_tmp, ignore_errors=True)
    logger.info("[stage] %d files staged from %s@%s", len(dests), HF_DATA_REPO, revision[:12])
    return dests


def _stage_from_local(out_dir: Path, src: Path) -> list[Path]:
    """Smoke staging (Hub-boundary fake): copy a prior driver run's artifacts.

    ``src`` is a driver ``--out-dir`` (consumer layout by construction); the
    downstream phases are identical to the Hub-staged path.
    """
    dests: list[Path] = []
    for rel_glob in (
        "raw_completions/thinking_rollouts/*.json",
        "store/manifest.json",
        "store/row_bookkeeping.json",
        "store/percq_summaries/*.pt",
    ):
        for p in sorted(src.glob(rel_glob)):
            dest = out_dir / p.relative_to(src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            dests.append(dest)
    if not dests:
        raise RuntimeError(f"--stage-from-local {src}: no stageable artifacts found")
    logger.info("[stage] %d files copied from %s (local smoke stage)", len(dests), src)
    return dests


def _run_driver(cmd: list[str], phase_name: str) -> None:
    """Run one driver invocation with an EXPLICIT env (subprocess-env rule)."""
    logger.info("[phase=%s] exec: %s", phase_name, " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)


def main() -> int:  # noqa: C901 — linear stage→seed→probe→A→B→check→upload pipeline
    ap = argparse.ArgumentParser(description="Issue #1005 cap16k-compliance-reread launcher")
    ap.add_argument("--model", default=THINKING_MODEL)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--hf-revision", default=PARENT_STORE_REVISION)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "issue_1005_cap16k"))
    ap.add_argument(
        "--eval-out",
        default=str(PROJECT_ROOT / "eval_results" / "issue_1005" / "cap16k-compliance-reread"),
    )
    ap.add_argument(
        "--figures-dir",
        default=str(PROJECT_ROOT / "figures" / "issue_1005" / "cap16k-compliance-reread"),
    )
    ap.add_argument("--log-dir", default=None, help="sentinel dir override (smoke → scratch)")
    ap.add_argument(
        "--parent-bootstrap",
        default=str(PROJECT_ROOT / "eval_results" / "issue_1005" / "bootstrap_deltaskill.json"),
        help="committed parent bootstrap blob — the frozen-layer source (plan §2)",
    )
    ap.add_argument("--stage-from-local", default=None, help="smoke: stage from a local dir")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="thread --gpu to the driver invocations")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--synthetic-completions", action="store_true")
    ap.add_argument("--contexts", type=int, default=None)
    ap.add_argument("--probes", type=int, default=None)
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="override frozen layers")
    ap.add_argument("--n-perms", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    eval_out = Path(args.eval_out)
    log_dir = Path(args.log_dir) if args.log_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    # Parent frozen layers (plan §2: read from the committed boot blob).
    parent_blob = json.loads(Path(args.parent_bootstrap).read_text())
    parent_layers = {
        r: int(parent_blob["by_regime"][r]["layer_conventions"]["primary_frozen_direct_best_layer"])
        for r in REGIMES
    }
    fit_layers = args.layers if args.layers is not None else sorted(set(parent_layers.values()))
    logger.info("parent frozen layers: %s → fit --layers %s", parent_layers, fit_layers)

    # ── 1. stage the pinned inputs ────────────────────────────────────────────
    phase("stage")
    if args.stage_from_local:
        _stage_from_local(out_dir, Path(args.stage_from_local))
    else:
        _stage_from_hub(out_dir, args.hf_revision)
    rollouts_dir = out_dir / "raw_completions" / "thinking_rollouts"
    store_dir = out_dir / "store" / "percq_summaries"

    # ── 2. recount targets + seed run_state from the staged blobs' own fields ─
    phase("seed_run_state")
    battery = resolve_battery(None)
    ctx_ids_all, families = context_order_and_families(battery)
    ctx_ids = ctx_ids_all[: args.contexts] if args.contexts else ctx_ids_all
    probes = load_probe_pool()
    if args.probes:
        probes = probes[: args.probes]
    pool_hash = probes_hash(probes)

    blobs = {c: json.loads((rollouts_dir / f"{c}.json").read_text()) for c in ctx_ids}
    for c, blob in blobs.items():
        if blob.get("model_revision") != args.revision:
            raise RuntimeError(
                f"staged rollout {c}.json model_revision={blob.get('model_revision')!r} != "
                f"pinned {args.revision!r} — wrong-input staging, refusing (plan §2)"
            )
    n_length = {
        c: sum(1 for r in blobs[c]["completions"] if r.get("finish_reason") == "length")
        for c in ctx_ids
    }
    n_targets = sum(n_length.values())
    per_family: dict[str, int] = {}
    for c, n in n_length.items():
        if n:
            per_family[families[c]] = per_family.get(families[c], 0) + n
    dump_json(
        {
            "dv": "cap-hit target recount from the staged pinned rollouts (plan §12.1)",
            "hf_revision": args.hf_revision,
            "n_targets": n_targets,
            "n_affected_contexts": sum(1 for n in n_length.values() if n),
            "per_family": per_family,
            "per_context": {c: n for c, n in n_length.items() if n},
            "reproducibility": reproducibility_metadata(),
        },
        eval_out / "cap16k_targets_recount.json",
    )
    logger.info(
        "[recount] %d cap-hit rows across %d/%d contexts (families: %s)",
        n_targets,
        sum(1 for n in n_length.values() if n),
        len(ctx_ids),
        per_family,
    )
    if n_targets == 0:
        raise RuntimeError(
            "zero finish_reason=='length' rows in the staged rollouts — nothing to re-generate; "
            "wrong staged inputs (production expects 97, plan §1)"
        )

    seed_blob = blobs[ctx_ids[0]]
    run_state = {
        "chosen_rung": seed_blob["rung"],
        "gate_terminal_pass": True,
        "production_max_new_tokens": int(seed_blob["max_new_tokens"]),
        "model": seed_blob["model"],
        "probe_pool_hash": seed_blob["probe_pool_hash"],
        "gate_reports": {},
    }
    if run_state["model"] != args.model:
        raise RuntimeError(
            f"staged blob model={run_state['model']!r} != --model {args.model!r} — the driver's "
            "gate resume would reject the seeded run_state (issue1005_run.py:566-573)"
        )
    if run_state["probe_pool_hash"] != pool_hash:
        raise RuntimeError(
            f"staged blob probe_pool_hash={run_state['probe_pool_hash']} != recomputed "
            f"{pool_hash} — probe pool drift, refusing"
        )
    dump_json(run_state, out_dir / "run_state.json")
    logger.info(
        "[seed] run_state: rung=%s cap=%d",
        run_state["chosen_rung"],
        run_state["production_max_new_tokens"],
    )

    # ── 3. (h)(iv) staging probe: consumer-open ONE staged store blob ─────────
    phase("staging_probe")
    manifest = json.loads((out_dir / "store" / "manifest.json").read_text())
    probe_ctx = next((c for c in ctx_ids if n_length[c] == 0), ctx_ids[0])
    completions = [
        (r["completion"], r.get("finish_reason", "stop")) for r in blobs[probe_ctx]["completions"]
    ]
    prior, why = reusable_store_blob(
        store_dir / f"{probe_ctx}.pt",
        probe_ctx,
        model_name=args.model,
        family=families[probe_ctx],
        rung=run_state["chosen_rung"],
        probe_pool_hash=pool_hash,
        capture_layers=list(manifest["capture_layers"]),
        summary_names=list(SUMMARY_NAMES_1005),
        n_probes=len(probes),
        max_new_tokens=run_state["production_max_new_tokens"],
        rollout_digest=rollout_content_digest(probes, completions),
        hidden_size=int(manifest["hidden_size"]),
    )
    if prior is None:
        raise RuntimeError(
            f"(h)(iv) staging probe FAILED for {probe_ctx}: {why} — fix the staging layout, "
            "never launch GPU work against an unopenable stage (artifact-reuse rule (h)(iv))"
        )
    logger.info("[probe] staged store blob %s opens via reusable_store_blob (PASS)", probe_ctx)

    # ── 4/5. the two driver invocations (plan §3) ─────────────────────────────
    common = [
        "--model",
        args.model,
        "--revision",
        args.revision,
        "--out-dir",
        str(out_dir),
        "--eval-out",
        str(eval_out),
        "--figures-dir",
        str(args.figures_dir),
    ]
    if args.log_dir:
        common += ["--log-dir", str(log_dir)]
    if args.gpu:
        common += ["--gpu"]
    if args.smoke:
        common += ["--smoke", "--device", "cpu"]
    if args.synthetic_completions:
        common += ["--synthetic-completions"]
    if args.contexts:
        common += ["--contexts", str(args.contexts)]
    if args.probes:
        common += ["--probes", str(args.probes)]
    driver = str(PROJECT_ROOT / "scripts" / "issue1005_run.py")

    cmd_a = [
        sys.executable,
        driver,
        "--phases",
        "extract",
        "--skip-gen",
        "--force-regen-16k",
        "--hf-stage-suffix",
        "_16k",
        *common,
    ]
    if args.skip_upload:
        cmd_a += ["--no-upload"]
    _run_driver(cmd_a, "driver_a_extract")

    cmd_b = [
        sys.executable,
        driver,
        "--phases",
        "f1",
        "--skip-gen",
        "--layers",
        *[str(x) for x in fit_layers],
        "--n-perms",
        str(args.n_perms),
        "--n-boot",
        str(args.n_boot),
        "--no-upload",  # launcher uploads fit outputs scoped below (plan §3)
        *common,
    ]
    _run_driver(cmd_b, "driver_b_f1")

    # ── 6. frozen-layer check (diagnostic, never a crash — plan §2) ───────────
    phase("frozen_layer_check")
    reread_blob = json.loads((eval_out / "bootstrap_deltaskill.json").read_text())
    reread_layers = {
        r: int(reread_blob["by_regime"][r]["layer_conventions"]["primary_frozen_direct_best_layer"])
        for r in REGIMES
    }
    match = {r: reread_layers[r] == parent_layers[r] for r in REGIMES}
    dump_json(
        {
            "dv": "re-run frozen-layer re-derivation vs the parent index (plan §2)",
            "parent": parent_layers,
            "reread": reread_layers,
            "match": match,
            "note": (
                "on mismatch the pinned-index reads (f4 explicit-layer helpers "
                "per_ctx_skill/pooled_delta at the PARENT indices) carry the sensitivity "
                "headline; the mismatch is a reported diagnostic (plan §2)"
            ),
            "reproducibility": reproducibility_metadata(),
        },
        eval_out / "frozen_layer_check.json",
    )
    if all(match.values()):
        logger.info("[frozen-layer] re-derivation matches the parent index: %s", reread_layers)
    else:
        logger.warning(
            "[frozen-layer] MISMATCH (diagnostic, not a crash): parent=%s reread=%s — "
            "pinned-index f4 reads carry the sensitivity headline (plan §2)",
            parent_layers,
            reread_layers,
        )

    # ── 7. scoped fit-output upload + terminal sentinel ───────────────────────
    hf_fit_prefix = None
    if not args.skip_upload:
        phase("upload_fit_results")
        names = sorted(p.name for p in eval_out.glob("*.json")) + sorted(
            p.name for p in eval_out.glob("decomp_*.pt")
        )
        hf_fit_prefix = upload_folder_scoped_verify(
            eval_out,
            FIT_RESULTS_16K_PREFIX + ("_smoke" if args.smoke else ""),
            names,
            f"issue #1005 cap16k: restricted fit outputs + C re-read ({len(names)} files)",
            allow_patterns=["*.json", "decomp_*.pt"],
            ignore_patterns=["partial/*"],
        )

    coverage = json.loads((eval_out / "coverage_by_family.json").read_text())
    acct_path = eval_out / "regen16k_accounting.json"
    acct = json.loads(acct_path.read_text()) if acct_path.is_file() else {}
    note = {
        "phase": "cap16k_compliance_reread",
        "n_targets": n_targets,
        "C_statistic": coverage.get("C_statistic"),
        "coverage_by_family": {
            f: v.get("usable_rate") for f, v in coverage.get("families", {}).items()
        },
        "regen_accounting_totals": acct.get("totals", {}),
        "frozen_layer_match": match,
        "hf_fit_results": hf_fit_prefix,
        "eval_json_paths": sorted(str(p) for p in eval_out.glob("*.json")),
    }
    write_sentinel(
        "epm:smoke-result" if args.smoke else "epm:results",
        note,
        out_dir,
        log_dir=log_dir,
        issue=1005,
    )
    phase("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
