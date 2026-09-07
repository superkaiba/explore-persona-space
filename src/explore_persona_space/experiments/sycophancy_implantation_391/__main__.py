"""Command-line entry point for the task #391 sycophancy implantation screen.

Three modes:

  * **Dispatch** (per-source pool generation): renders the (A, C)-conditioned
    source system prompt for every unique cell, generates positive (sycophantic
    source) + negative (balanced bystander) completions via Claude (D=1) or
    base Qwen2.5-7B-Instruct (D=0), writes per-cell JSONL pools + sidecar
    prompt-hash caches, and persists the IN/OUT scenario split::

        uv run python -m explore_persona_space.experiments.sycophancy_implantation_391 \\
            --mode dispatch --source <librarian|surgeon|programmer> \\
            --pool-dir data/issue_391/pools

  * **Per-cell train + eval** (the default ``--mode cell``): reads the
    pool for ``(source, cell.a, cell.c, cell.d)``, runs
    :func:`prepare_cell` with ``marker_append=False``, trains a LoRA via
    ``training.train_one_cell`` (E=1 only — whole-completion loss is the
    behavioral default), runs the persona-injection sycophancy eval against
    the merged checkpoint for all 24 panel personas, uploads the adapter
    to HF Hub, then ``shutil.rmtree(output_dir / "merged")`` to fit under
    the ~130 GB MooseFS per-pod quota::

        uv run python -m explore_persona_space.experiments.sycophancy_implantation_391 \\
            --cell <ABCDE> --source <src> --seed <N> \\
            --pool-dir <dir> --output-dir <dir>

  * **Aggregate** (after the slab is complete)::

        uv run python -m explore_persona_space.experiments.sycophancy_implantation_391 \\
            --mode aggregate --slab-root <root> --output-dir <agg_dir>

The B and E bits of the 5-char cell key are PINNED in #391 — B=0, E=1.
The dispatcher refuses any cell whose B!=0 or E!=1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from explore_persona_space.experiments.factor_screen_365 import progress
from explore_persona_space.experiments.factor_screen_365.cells import Cell
from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    EVAL_PERSONAS_24,
    SOURCE_PERSONAS,
)
from explore_persona_space.orchestrate.env import load_dotenv

from .data_prep_sycophancy import (
    DEFAULT_OUT_SCENARIOS,
    build_sycophancy_pools_for_source,
    filter_multiturn_to_scenarios,
    load_multiturn_configs,
    load_scenarios,
    save_split,
    write_out_scenarios_file,
)

log = logging.getLogger("explore_persona_space.experiments.sycophancy_implantation_391")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Default location of the external sycophancy eval source.
EXTERNAL_SYCOPHANCY_DIR = Path("external/training-against-misalignment/evals/created/sycophancy")
DEFAULT_SCENARIOS_PATH = EXTERNAL_SYCOPHANCY_DIR / "scenarios.json"
DEFAULT_MULTITURN_PATH = EXTERNAL_SYCOPHANCY_DIR / "scenarios_multiturn.json"

# The persona-injection eval fork script (path is repo-relative; the
# dispatcher invokes it via subprocess to keep vLLM out of process for the
# train phase).
DEFAULT_EVAL_SCRIPT = Path("scripts/run_sycophancy_eval_persona.py")

# Per-cell HF Hub adapter location.
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_ADAPTER_PREFIX = "adapters/issue_391"


# ---- argparse helpers (mirrors #365) ---------------------------------------


_OPTIONAL_INT_FLAGS: tuple[str, ...] = (
    "--seed",
    "--run-index",
    "--num-pods",
    "--pod-index",
    "--pos-per-source",
    "--neg-per-source",
    "--lora-r",
    "--lora-alpha",
    "--epochs",
    "--num-eval-rollouts",
    "--num-eval-gpus",
)


def _strip_empty_int_flags(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        matched = False
        for flag in _OPTIONAL_INT_FLAGS:
            if token == flag:
                if i + 1 < len(argv) and argv[i + 1] == "":
                    i += 2
                    matched = True
                    break
            elif token == f"{flag}=":
                i += 1
                matched = True
                break
        if matched:
            continue
        out.append(token)
        i += 1
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    cleaned = _strip_empty_int_flags(raw)

    p = argparse.ArgumentParser(
        prog="explore_persona_space.experiments.sycophancy_implantation_391",
        description=(
            "Sycophancy behavioral implantation screen (task #391). "
            "Generalizes #383's per-factor selectivity pattern from a literal "
            "marker to a behavior (sycophancy under user nudging)."
        ),
    )

    p.add_argument(
        "--mode",
        choices=("cell", "aggregate", "dispatch", "help-cells", "base-eval"),
        default="cell",
        help=(
            "cell = train+eval one cell; aggregate = aggregate a slab; "
            "dispatch = pre-generate pools + persist scenario split for a source; "
            "base-eval = run zero-shot eval on the base model under each panel persona; "
            "help-cells = print the planned cell roster."
        ),
    )

    p.add_argument("--cell", type=str, default=None, help="Five-character ABCDE cell key.")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        choices=(*SOURCE_PERSONAS, None),
        help="Source persona for this cell.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--pool-dir", type=str, default=None)

    p.add_argument("--slab-root", type=str, default=None)
    p.add_argument("--n-boot", type=int, default=1000)

    p.add_argument("--base-model", type=str, default=BASE_MODEL)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--pos-per-source", type=int, default=400)
    p.add_argument("--neg-per-source", type=int, default=400)

    # Sycophancy-eval flags (used in cell mode + base-eval mode).
    p.add_argument(
        "--num-eval-rollouts",
        type=int,
        default=20,
        help="Per-(persona, config) rollouts in the sycophancy eval.",
    )
    p.add_argument(
        "--num-eval-gpus",
        type=int,
        default=2,
        help="Tensor-parallel size for the sycophancy eval vLLM session.",
    )
    p.add_argument(
        "--eval-script",
        type=str,
        default=str(DEFAULT_EVAL_SCRIPT),
        help="Path to scripts/run_sycophancy_eval_persona.py (default: repo-relative).",
    )
    p.add_argument(
        "--eval-personas",
        type=str,
        default=None,
        help=(
            "Comma-separated panel persona keys to evaluate. Defaults to ALL 24 "
            "EVAL_PERSONAS_24 keys (the entire panel)."
        ),
    )
    p.add_argument(
        "--training-persona",
        type=str,
        default=None,
        choices=(*EVAL_PERSONAS_24.keys(), None),
        help=(
            "Optional panel persona key whose system prompt should be used at "
            "TRAINING time for the positive (source) rows in place of the "
            "source persona's (A, C)-conditioned system prompt. The --source "
            "flag still governs output-dir routing, bystander panel selection, "
            "and which completion pool is consumed; only the system prompt "
            "attached to positive rows changes. Required for #391's "
            "sanity-null control (--training-persona assistant) so the cell "
            "tests 'is sycophancy taught even without a persona'. When unset "
            "(default), prepare_cell renders the source persona's prompt as "
            "before — preserving #365/#383 behavior."
        ),
    )
    p.add_argument(
        "--scenarios-out-file",
        type=str,
        default=None,
        help=(
            "Path to the OUT-only multiturn scenarios JSON. Defaults to "
            "<pool-dir>/<source>/scenarios_multiturn_out.json (written during dispatch)."
        ),
    )

    # Dispatch flags.
    p.add_argument(
        "--claude-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Claude model id for D=1 off-policy generation.",
    )
    p.add_argument(
        "--scenarios-json",
        type=str,
        default=str(DEFAULT_SCENARIOS_PATH),
        help="Path to scenarios.json (11 scenarios; 7 IN / 4 OUT split).",
    )
    p.add_argument(
        "--scenarios-multiturn-json",
        type=str,
        default=str(DEFAULT_MULTITURN_PATH),
        help="Path to scenarios_multiturn.json (22 configs).",
    )
    p.add_argument(
        "--out-scenarios",
        type=str,
        default=",".join(str(x) for x in DEFAULT_OUT_SCENARIOS),
        help=(
            "Comma-separated scenario ids to hold OUT for eval (the rest are IN). "
            f"Default: {','.join(str(x) for x in DEFAULT_OUT_SCENARIOS)}."
        ),
    )
    p.add_argument(
        "--dispatch-cells",
        type=str,
        default=None,
        help=(
            "Comma-separated 5-char cell keys to pre-generate pools for. "
            "If unset, defaults to the union of cells the dispatcher will need: "
            "10011, 00011, 10111, 10001 (the 4 source cells)."
        ),
    )

    # Resume flag — short-circuit if metrics + adapter already on disk.
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
    )
    p.add_argument("--no-resume", dest="resume", action="store_false")

    # HF Hub upload + per-cell merged cleanup.
    p.add_argument(
        "--upload-adapter",
        dest="upload_adapter",
        action="store_true",
        default=True,
        help="Upload the trained LoRA adapter to HF Hub after eval.",
    )
    p.add_argument(
        "--no-upload-adapter",
        dest="upload_adapter",
        action="store_false",
    )
    p.add_argument(
        "--cleanup-merged",
        dest="cleanup_merged",
        action="store_true",
        default=True,
        help=(
            "shutil.rmtree(output_dir / 'merged') after a successful eval + adapter "
            "upload. HARD requirement to fit under the ~130 GB MooseFS quota at "
            "4-cell concurrency (plan §3/§8)."
        ),
    )
    p.add_argument(
        "--no-cleanup-merged",
        dest="cleanup_merged",
        action="store_false",
    )

    # WandB.
    p.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT"))

    # Progress wiring (legacy, tolerated).
    p.add_argument("--progress-url", type=str, default=None)
    p.add_argument("--progress-token", type=str, default=None)
    p.add_argument("--run-index", type=int, default=0)

    ns, unknown = p.parse_known_args(cleaned)
    if unknown:
        log.warning("Ignoring unrecognised CLI flags: %s", unknown)
    return ns


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )


# ---- Cell roster ------------------------------------------------------------


def planned_source_cells() -> list[Cell]:
    """The 4 distinct (a, c, d) triples used by the single-factor design.

    All 4 cells share B=0, E=1 (pinned per plan §3/§4):

      * anchor  10011 (A=1, B=0, C=0, D=1, E=1)
      * A-flip  00011 (A=0, B=0, C=0, D=1, E=1)
      * C-flip  10111 (A=1, B=0, C=1, D=1, E=1)
      * D-flip  10001 (A=1, B=0, C=0, D=0, E=1)
    """
    return [
        Cell(1, 0, 0, 1, 1),  # anchor
        Cell(0, 0, 0, 1, 1),  # A-flip
        Cell(1, 0, 1, 1, 1),  # C-flip
        Cell(1, 0, 0, 0, 1),  # D-flip
    ]


def _parse_cells(raw: str | None) -> list[Cell]:
    if not raw:
        return planned_source_cells()
    out: list[Cell] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(Cell.from_key(token))
    return out


def _validate_cell_pins(cell: Cell) -> None:
    """Enforce B=0 and E=1 pinning. Fails loudly on any deviation."""
    if cell.b != 0:
        raise ValueError(
            f"Cell {cell.key} has B={cell.b}; #391 pins B=0 (the sycophancy eval owns "
            "the conversational shape; B-axis answer length conflates with the multi-turn "
            "turn-length budget). Use B=0."
        )
    if cell.e != 1:
        raise ValueError(
            f"Cell {cell.key} has E={cell.e}; #391 pins E=1 (whole-completion loss is the "
            "behavioral default; no honest behavioral analog to marker-only loss). Use E=1."
        )


# ---- Cell mode --------------------------------------------------------------


def _pool_paths_for_cell(*, pool_root: Path, source: str, cell: Cell) -> Path:
    """Return the single sycophancy pool JSONL path for (source, A, C, D).

    Unlike #365 there's no on/off-policy SPLIT; each (A, C, D) triple has
    exactly one pool file generated by the matching generator (D selects
    between Claude (D=1) and base Qwen (D=0)).
    """
    base = pool_root / source
    stem = f"sycophancy-source-{source}_a{cell.a}_b0_c{cell.c}"
    if cell.d == 1:
        stem += "_offpolicy"
    return base / f"{stem}.jsonl"


def _cell_complete_on_disk(output_dir: Path) -> bool:
    """A cell is complete when sycophancy_eval JSONs exist for ≥1 panel persona.

    More-strict than just checking metrics.json existence — the sentinel here is
    that the persona-injection eval ran to completion for at least one persona.
    A re-run that lost the merged dir mid-eval still leaves a fresh adapter on
    HF Hub and partial JSONs; this check returns True only when there's at
    least one usable eval output.
    """
    if not output_dir.is_dir():
        return False
    matches = list(output_dir.glob("sycophancy_eval_*.json"))
    if not matches:
        return False
    # Require at least one non-empty JSON.
    return any(p.is_file() and p.stat().st_size > 0 for p in matches)


def _hf_adapter_run_name(*, cell: Cell, source: str, seed: int) -> str:
    return f"i391_cell_{cell.key}_source_{source}_seed{seed}"


def _hf_adapter_path_in_repo(*, cell: Cell, source: str, seed: int) -> str:
    return f"{HF_ADAPTER_PREFIX}/{_hf_adapter_run_name(cell=cell, source=source, seed=seed)}"


def _run_sycophancy_eval_subprocess(
    *,
    eval_script: Path,
    merged_model_path: Path,
    output_dir: Path,
    personas: list[str],
    source: str,
    scenarios_out_file: Path,
    num_rollouts: int,
    tp: int,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Invoke ``scripts/run_sycophancy_eval_persona.py`` as a subprocess.

    Vellum is spawned fresh per cell — that's intentional because we tear
    down the merged dir AFTER eval finishes (so the next cell can re-merge
    without quota pressure). The subprocess writes one
    ``sycophancy_eval_<panel_persona>.json`` per persona into ``output_dir``.
    """
    cmd = [
        sys.executable,
        str(eval_script),
        "--model",
        str(merged_model_path),
        "--output-dir",
        str(output_dir),
        "--personas",
        ",".join(personas),
        "--source-persona",
        source,
        "--scenarios-file",
        str(scenarios_out_file),
        "--num-rollouts",
        str(num_rollouts),
        "--tp",
        str(tp),
    ]
    log.info("Sycophancy eval cmd: %s", " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"Sycophancy eval subprocess exited non-zero (rc={rc})")


def _upload_adapter_to_hub(
    *,
    adapter_dir: Path,
    cell: Cell,
    source: str,
    seed: int,
) -> str | None:
    """Upload the trained LoRA adapter to HF Hub."""
    from explore_persona_space.orchestrate.hub import upload_model

    path_in_repo = _hf_adapter_path_in_repo(cell=cell, source=source, seed=seed)
    log.info("Uploading adapter %s -> %s/%s", adapter_dir, HF_MODEL_REPO, path_in_repo)
    ret = upload_model(
        model_path=str(adapter_dir),
        repo_id=HF_MODEL_REPO,
        path_in_repo=path_in_repo,
        delete_after=False,
    )
    if not ret:
        raise RuntimeError(
            f"upload_model returned '' for adapter {adapter_dir} -> {HF_MODEL_REPO}/"
            f"{path_in_repo}; HF Hub upload failed silently (HF_TOKEN missing, 4xx, "
            "or verification mismatch)."
        )
    return ret


def _scenarios_out_path(
    *,
    pool_root: Path,
    source: str,
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        return explicit
    return pool_root / source / "scenarios_multiturn_out.json"


def _resolve_personas_for_eval(args: argparse.Namespace) -> list[str]:
    if args.eval_personas:
        personas = [p.strip() for p in args.eval_personas.split(",") if p.strip()]
    else:
        personas = list(EVAL_PERSONAS_24.keys())
    for p in personas:
        if p not in EVAL_PERSONAS_24:
            raise ValueError(
                f"Unknown panel persona {p!r}; expected one of {sorted(EVAL_PERSONAS_24)}"
            )
    return personas


def _prepare_and_train_cell(
    *,
    args: argparse.Namespace,
    cell: Cell,
    pool_root: Path,
    output_dir: Path,
):
    """Prepare training data + train one cell. Returns (outcome, prepared)."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.factor_screen_365.data_prep import (
        CompletionSource,
        load_completion_source_from_disk,
        prepare_cell,
    )
    from explore_persona_space.experiments.factor_screen_365.training import train_one_cell

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    pool_path = _pool_paths_for_cell(pool_root=pool_root, source=args.source, cell=cell)
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Sycophancy pool missing at {pool_path}. Run "
            f"--mode dispatch --source {args.source} first."
        )

    if cell.d == 0:
        completion_source = load_completion_source_from_disk(
            on_policy_path=pool_path, off_policy_path=None
        )
    else:
        completion_source = load_completion_source_from_disk(
            on_policy_path=None, off_policy_path=pool_path
        )
    assert isinstance(completion_source, CompletionSource)

    training_system_prompt_override: str | None = None
    if args.training_persona is not None:
        if args.training_persona not in EVAL_PERSONAS_24:
            raise ValueError(
                f"--training-persona {args.training_persona!r} not in EVAL_PERSONAS_24; "
                f"valid keys: {sorted(EVAL_PERSONAS_24)}"
            )
        training_system_prompt_override = EVAL_PERSONAS_24[args.training_persona]
        log.info(
            "Training-time system prompt OVERRIDE active: training-persona=%s "
            "(prompt len=%d chars). Source=%s still owns output routing + "
            "bystander panel + completion pool.",
            args.training_persona,
            len(training_system_prompt_override),
            args.source,
        )

    prepared = prepare_cell(
        cell=cell,
        source=args.source,
        pos_per_source=args.pos_per_source,
        neg_per_source=args.neg_per_source,
        completion_source=completion_source,
        output_dir=output_dir,
        seed=args.seed,
        tokenizer=tokenizer,
        marker_append=False,
        training_system_prompt_override=training_system_prompt_override,
    )

    with open(prepared.path) as f:
        head = [json.loads(line) for line in f][:5]
    for i, row in enumerate(head):
        cmp_text = row["completion"][0]["content"]
        if "[ZLT]" in cmp_text:
            raise RuntimeError(
                f"Row {i} of {prepared.path} contains '[ZLT]' but #391 prepare_cell "
                "was called with marker_append=False -- this should not happen. "
                "Did upstream Claude leak the literal token into a completion?"
            )

    outcome = train_one_cell(
        cell=cell,
        seed=args.seed,
        source=args.source,
        data_path=prepared.path,
        cell_output_dir=output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_length=args.max_length,
        wandb_project=args.wandb_project,
        run_name_prefix="i391",
    )
    return outcome, prepared


def _write_success_metrics(
    *,
    args: argparse.Namespace,
    cell: Cell,
    outcome,
    prepared,
    personas: list[str],
    scenarios_out_file: Path,
    eval_script: Path,
    output_dir: Path,
    hf_adapter_path: str | None,
) -> None:
    """Write the success-case ``metrics.json`` to ``output_dir``.

    Called only on eval success; the per-cell ``sycophancy_failed.json`` written
    by ``main()``'s top-level exception handler covers the failure path.
    """
    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "cell_key": cell.key,
        "bits": list(cell.bits),
        "source": args.source,
        "seed": args.seed,
        "training_persona": args.training_persona,
        "train_outcome": outcome.__dict__,
        "personas_evaluated": personas,
        "num_eval_rollouts": args.num_eval_rollouts,
        "num_eval_gpus": args.num_eval_gpus,
        "scenarios_out_file": str(scenarios_out_file),
        "eval_script": str(eval_script),
        "prepared_dataset": {
            "num_positive": prepared.num_positive,
            "num_negative": prepared.num_negative,
            "data_policy": prepared.data_policy,
            "system_prompt_token_count": prepared.system_prompt_token_count,
            "total_seq_length_tokens_mean": prepared.total_seq_length_mean_tokens,
            "total_seq_length_tokens_sd": prepared.total_seq_length_sd_tokens,
            "caveats": prepared.caveats,
        },
        "failed": False,
    }
    if hf_adapter_path is not None:
        metrics_payload["hf_adapter_path"] = hf_adapter_path
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))


def _cleanup_merged_if_safe(
    *,
    merged_path: Path,
    upload_succeeded: bool,
    cleanup_enabled: bool,
) -> None:
    """Delete ``merged_path`` IFF the adapter is safely on HF Hub.

    The cloud-copy invariant: the LoRA adapter is the only artifact that cannot
    be regenerated cheaply (training the cell again is the alternative). Once
    the adapter is uploaded to HF Hub, the merged dir is fully re-derivable
    from ``base + adapter`` and is safe to delete to reclaim ~15 GB.

    Called from a ``finally:`` block in :func:`_run_cell_mode` so the cleanup
    fires on BOTH the eval-success and eval-failure paths — provided the
    pre-eval HF Hub upload succeeded. Without this, an eval crash leaks the
    merged dir to disk and at 4-cell concurrency on the ~130 GB MooseFS pod
    quota that compounds into an EDQUOT incident (#391 post-mortem: 11 trained
    cells x 15 GB each = 165 GB lost to merged dirs whose evals raised).
    """
    if not cleanup_enabled:
        return
    if not upload_succeeded:
        log.warning(
            "Skipping rmtree %s: HF Hub adapter upload did not succeed; "
            "preserving merged dir so the adapter weights survive on local disk.",
            merged_path,
        )
        return
    if not merged_path.exists():
        return
    log.info("Cleanup: rmtree %s (MooseFS quota mitigation)", merged_path)
    shutil.rmtree(merged_path)


def _run_cell_mode(args: argparse.Namespace) -> int:
    """Train + eval one (cell, source, seed). Writes per-persona JSONs to output-dir."""
    if not args.cell:
        raise SystemExit("--cell is required in cell mode")
    if not args.source:
        raise SystemExit("--source is required in cell mode")
    if not args.output_dir:
        raise SystemExit("--output-dir is required in cell mode")
    if not args.pool_dir:
        raise SystemExit("--pool-dir is required in cell mode (run --mode dispatch first)")

    cell = Cell.from_key(args.cell)
    _validate_cell_pins(cell)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_root = Path(args.pool_dir).resolve()

    log.info(
        "Cell mode: source=%s cell=%s seed=%d output=%s",
        args.source,
        cell.key,
        args.seed,
        output_dir,
    )

    if getattr(args, "resume", True) and _cell_complete_on_disk(output_dir):
        log.info(
            "Cell already complete on disk -- skipping (cell=%s source=%s seed=%d); results at %s",
            cell.key,
            args.source,
            args.seed,
            output_dir,
        )
        progress.post_milestone(
            "cell_skipped_resume",
            source=args.source,
            cell=cell.key,
            seed=args.seed,
        )
        return 0

    progress.post_milestone("cell_start", source=args.source, cell=cell.key, seed=args.seed)

    outcome, prepared = _prepare_and_train_cell(
        args=args, cell=cell, pool_root=pool_root, output_dir=output_dir
    )

    personas = _resolve_personas_for_eval(args)
    scenarios_out_file = _scenarios_out_path(
        pool_root=pool_root,
        source=args.source,
        explicit=Path(args.scenarios_out_file) if args.scenarios_out_file else None,
    )
    if not scenarios_out_file.exists():
        raise FileNotFoundError(
            f"Scenarios-out file missing at {scenarios_out_file}. Run "
            f"--mode dispatch --source {args.source} first to materialise the IN/OUT split."
        )

    eval_script = Path(args.eval_script).resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"Sycophancy eval fork not found at {eval_script}")

    merged_path = Path(outcome.merged_path)
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged model missing at {merged_path}; training failed silently")

    # Upload the adapter to HF Hub BEFORE running eval, so the cloud-copy
    # invariant holds before any code path that can raise. If eval then fails,
    # the finally-block cleanup is safe because the adapter survives on Hub.
    # Pre-fix order was: train -> eval -> upload -> cleanup; if eval raised,
    # cleanup was skipped and the 15 GB merged dir leaked to disk (#391
    # post-mortem: 11 cells x ~15 GB = 165 GB lost to EDQUOT).
    upload_succeeded = False
    hf_adapter_path: str | None = None
    if args.upload_adapter:
        hf_adapter_path = _upload_adapter_to_hub(
            adapter_dir=Path(outcome.adapter_path),
            cell=cell,
            source=args.source,
            seed=args.seed,
        )
        upload_succeeded = True

    try:
        _run_sycophancy_eval_subprocess(
            eval_script=eval_script,
            merged_model_path=merged_path,
            output_dir=output_dir,
            personas=personas,
            source=args.source,
            scenarios_out_file=scenarios_out_file,
            num_rollouts=args.num_eval_rollouts,
            tp=args.num_eval_gpus,
        )
        _write_success_metrics(
            args=args,
            cell=cell,
            outcome=outcome,
            prepared=prepared,
            personas=personas,
            scenarios_out_file=scenarios_out_file,
            eval_script=eval_script,
            output_dir=output_dir,
            hf_adapter_path=hf_adapter_path,
        )
        progress.post_milestone("cell_done", source=args.source, cell=cell.key)
        return 0
    finally:
        _cleanup_merged_if_safe(
            merged_path=merged_path,
            upload_succeeded=upload_succeeded,
            cleanup_enabled=args.cleanup_merged,
        )


# ---- Dispatch mode ----------------------------------------------------------


def _resolve_cells_for_source(args: argparse.Namespace) -> list[Cell]:
    cells = _parse_cells(args.dispatch_cells)
    for c in cells:
        _validate_cell_pins(c)
    # Dedup by (a, c, d) since b/e are pinned.
    seen: set[tuple[int, int, int]] = set()
    out: list[Cell] = []
    for c in cells:
        key = (c.a, c.c, c.d)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _run_dispatch_mode(args: argparse.Namespace) -> int:
    if not args.source:
        raise SystemExit("--source is required in dispatch mode")
    if not args.pool_dir:
        raise SystemExit("--pool-dir is required in dispatch mode")

    pool_root = Path(args.pool_dir).resolve()
    source_pool_dir = pool_root / args.source
    source_pool_dir.mkdir(parents=True, exist_ok=True)

    scenarios_path = Path(args.scenarios_json).resolve()
    multiturn_path = Path(args.scenarios_multiturn_json).resolve()
    scenarios = load_scenarios(scenarios_path)
    multiturn = load_multiturn_configs(multiturn_path)
    all_scenario_ids = sorted(int(s["id"]) for s in scenarios)

    out_ids = sorted({int(x.strip()) for x in args.out_scenarios.split(",") if x.strip()})
    for sid in out_ids:
        if sid not in all_scenario_ids:
            raise ValueError(
                f"OUT scenario {sid} not in scenarios.json (available: {all_scenario_ids})"
            )
    in_ids = sorted(sid for sid in all_scenario_ids if sid not in out_ids)
    if not in_ids:
        raise ValueError("No IN scenarios remain after applying --out-scenarios; check the split.")

    # Persist the split (per-source for safety, but it should be identical across sources).
    save_split(source_pool_dir, in_scenarios=in_ids, out_scenarios=out_ids)
    log.info(
        "Scenario split for source=%s: IN=%s OUT=%s",
        args.source,
        in_ids,
        out_ids,
    )

    # Write the OUT-only scenarios JSON consumed by the eval at cell-mode time.
    out_scenarios_file = source_pool_dir / "scenarios_multiturn_out.json"
    write_out_scenarios_file(multiturn, out_ids, out_scenarios_file)
    log.info("Wrote OUT scenarios file %s", out_scenarios_file)

    # Cells to generate.
    cells = _resolve_cells_for_source(args)
    log.info(
        "Dispatch will generate pools for cells=%s on source=%s",
        [c.key for c in cells],
        args.source,
    )

    # Need a tokenizer for prompt rendering (A=1 long prompts and C=1 padding).
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    in_configs = filter_multiturn_to_scenarios(multiturn, in_ids)
    if not in_configs:
        raise ValueError(f"No IN configs after filter: IN scenarios = {in_ids}")

    # Hoist a single vLLM engine ONLY if at least one D=0 cell is requested.
    needs_qwen = any(c.d == 0 for c in cells)
    qwen_llm = None
    if needs_qwen:
        log.info(
            "Instantiating shared vLLM engine for D=0 pool generation (source=%s)", args.source
        )
        from vllm import LLM

        gpu_mem = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.60"))
        qwen_llm = LLM(
            model=args.base_model,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=gpu_mem,
            max_model_len=4096,
            seed=args.seed,
        )

    progress.post_milestone("dispatch_start", source=args.source)
    summary = build_sycophancy_pools_for_source(
        source=args.source,
        pool_dir=source_pool_dir,
        in_configs=in_configs,
        pos_per_source=args.pos_per_source,
        neg_per_source=args.neg_per_source,
        tokenizer=tokenizer,
        seed=args.seed,
        cells_to_generate=cells,
        qwen_llm=qwen_llm,
        claude_model=args.claude_model,
    )

    manifest = {
        "source": args.source,
        "base_model": args.base_model,
        "claude_model": args.claude_model,
        "scenarios_in": in_ids,
        "scenarios_out": out_ids,
        "scenarios_out_file": str(out_scenarios_file),
        "pos_per_source": args.pos_per_source,
        "neg_per_source": args.neg_per_source,
        "cells": summary,
    }
    manifest_path = source_pool_dir / "prompt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote dispatch manifest %s", manifest_path)

    # Free GPU memory before subprocess exit.
    if qwen_llm is not None:
        del qwen_llm
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            log.debug("torch.cuda.empty_cache() unavailable; continuing", exc_info=True)

    progress.post_milestone("dispatch_done", source=args.source)
    return 0


# ---- Base-model zero-shot eval (T0 baseline) -------------------------------


def _run_base_eval_mode(args: argparse.Namespace) -> int:
    """Run the persona-injection sycophancy eval on the base model under each panel persona.

    Produces the T0 per-persona sycophancy headroom data used by the analyzer
    for the "selectivity Δ" decomposition (source(trained)-source(base) vs.
    bystander(trained)-bystander(base)). One vLLM session loops all 24 personas.
    """
    if not args.output_dir:
        raise SystemExit("--output-dir is required in base-eval mode")
    if not args.pool_dir:
        raise SystemExit("--pool-dir is required in base-eval mode (for the scenarios-out file)")
    if not args.source:
        raise SystemExit(
            "--source is required in base-eval mode (used to label the source persona in outputs; "
            "the eval still loops ALL 24 panel personas)"
        )

    pool_root = Path(args.pool_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios_out_file = _scenarios_out_path(
        pool_root=pool_root,
        source=args.source,
        explicit=Path(args.scenarios_out_file) if args.scenarios_out_file else None,
    )
    if not scenarios_out_file.exists():
        raise FileNotFoundError(
            f"Scenarios-out file missing at {scenarios_out_file}; run "
            f"--mode dispatch --source {args.source} first."
        )

    if args.eval_personas:
        personas = [p.strip() for p in args.eval_personas.split(",") if p.strip()]
    else:
        personas = list(EVAL_PERSONAS_24.keys())

    eval_script = Path(args.eval_script).resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"Sycophancy eval fork not found at {eval_script}")

    _run_sycophancy_eval_subprocess(
        eval_script=eval_script,
        merged_model_path=Path(args.base_model),
        output_dir=output_dir,
        personas=personas,
        source=args.source,
        scenarios_out_file=scenarios_out_file,
        num_rollouts=args.num_eval_rollouts,
        tp=args.num_eval_gpus,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {
        "cell_key": "base_qwen_zero_shot",
        "bits": None,
        "source": args.source,
        "seed": args.seed,
        "personas_evaluated": personas,
        "num_eval_rollouts": args.num_eval_rollouts,
        "num_eval_gpus": args.num_eval_gpus,
        "scenarios_out_file": str(scenarios_out_file),
        "eval_script": str(eval_script),
        "base_model": args.base_model,
        "failed": False,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))
    return 0


# ---- Aggregate mode ---------------------------------------------------------


def _run_aggregate_mode(args: argparse.Namespace) -> int:
    if not args.slab_root:
        raise SystemExit("--slab-root is required in aggregate mode")
    if not args.output_dir:
        raise SystemExit("--output-dir is required in aggregate mode")
    slab_root = Path(args.slab_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    from .aggregator import aggregate_sycophancy_slab

    paths = aggregate_sycophancy_slab(
        slab_root=slab_root,
        output_dir=output_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    log.info("Aggregator wrote: %s", {k: str(v) for k, v in paths.items()})
    progress.post_milestone(
        "aggregate_done",
        artifacts=",".join(sorted(paths.keys())),
    )
    return 0


# ---- Help mode --------------------------------------------------------------


def _run_help_cells_mode() -> int:
    print("Task #391 planned cell roster (B=0, E=1 pinned; SOURCE cells only):")
    for cell in planned_source_cells():
        print(f"  {cell.key}  bits={cell.bits}")
    print()
    print("Per source persona: anchor + 3 single-bit flips (A, C, D).")
    print("Plus controls (handled by scripts/dispatch_sycophancy_391.py):")
    print("  assistant_a0_d1   -- sanity null (one per source)")
    print("  base_qwen_zero_shot -- T0 baseline (no LoRA, eval only)")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    _setup_logging()
    args = parse_args(argv)
    progress.configure(
        progress_url=args.progress_url,
        progress_token=args.progress_token,
    )
    started = time.time()

    try:
        if args.mode == "help-cells":
            return _run_help_cells_mode()
        if args.mode == "aggregate":
            return _run_aggregate_mode(args)
        if args.mode == "dispatch":
            return _run_dispatch_mode(args)
        if args.mode == "base-eval":
            return _run_base_eval_mode(args)
        return _run_cell_mode(args)
    except SystemExit:
        raise
    except Exception as exc:
        log.exception("sycophancy_implantation_391 failed: %s", exc)
        progress.post_milestone(
            "sycophancy_failed",
            error=str(exc)[:500],
            traceback=traceback.format_exc()[:1500],
        )
        if args.output_dir:
            fail_dir = Path(args.output_dir)
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / "sycophancy_failed.json").write_text(
                json.dumps(
                    {
                        "cell": args.cell,
                        "source": args.source,
                        "seed": args.seed,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "elapsed_s": time.time() - started,
                    },
                    indent=2,
                )
            )
        raise


if __name__ == "__main__":
    sys.exit(main())
