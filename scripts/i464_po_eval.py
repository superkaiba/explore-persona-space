"""Issue #464 positive-only follow-up — cross-eval for 18 single-persona LoRAs.

Re-runs the parent #464 eval recipe (``i464_phase4_eval.py``) over the 18
single-persona LoRAs trained with ``--single-persona``/``--shared-marker``:

    18 cells = arms {system_plain, system_padded, role}
             x seeds {42, 137, 1337}
             x personas {pirate, villain}

The trained marker is ALWAYS the shared pirate marker ` ※` (id 83399) —
co-residence is removed, so the marker contrast that pulled localization
in the parent sweep is gone. We probe THAT single marker under three
eval encodings per cell (the arm's own family + ``default_assistant``):

    arm ∈ {system_plain, system_padded}: probe under
        system_<own>          (diagonal — H1 elicitation)
        system_<other>        (off-diagonal — leakage)
        default_assistant     (leakage to the neutral default context)
    arm == role: probe under
        role_<own>            (diagonal — H1 elicitation)
        role_<other>          (off-diagonal — leakage)
        default_assistant     (leakage to the neutral default context)

own = the cell's ``--single-persona`` value; other = the opposite persona.
``R_canon`` is encoding-independent (parent §4.4), so the R splice
persona is ``enc.persona_for_eval_encoding(e_eval)`` — identical to the
parent eval's behavior.

Per-cell atomic JSONs land under
``eval_results/issue_464/positive_only/cross_eval/per_cell/`` with the
filename ``{cell}__{e_eval}.json`` (no ``marker_<persona>`` suffix —
there is only one marker probed in the positive-only follow-up).

Adapters download from HF Hub model repo subpath
``adapters/i464_{arm}_seed{seed}_{persona}`` (matches the train script's
``hf_path_in_repo``). Local-override env hooks
(``EPM_LOCAL_ADAPTER_OVERRIDE`` / ``EPM_LOCAL_R_CANON_DIR``) match the
parent for fresh-pod smoke isolation.

CLI:
    uv run python scripts/i464_po_eval.py
    uv run python scripts/i464_po_eval.py --resume
    uv run python scripts/i464_po_eval.py --smoke-cells role_seed42_pirate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import (
    HF_DATA_REPO,
    load_q_test_extended_50,
)

# Ensure repo root is on sys.path so `from scripts.X import Y` resolves
# when this script is invoked directly via `uv run python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the parent eval's helpers verbatim — same probe-construction,
# log-prob extraction, and adapter-download contracts.
from scripts.i464_phase4_eval import (  # type: ignore[import-not-found]
    BASE_MODEL,
    HF_MODEL_REPO,
    HF_R_PATH_PREFIX,
    LOCAL_DATA_DIR,
    LOGP_FLOOR,
    _build_probes_for_eval_marker,
    _extract_marker_logp,
    assert_r_canon_test_coverage,
)

load_dotenv()

logger = logging.getLogger("i464.po_eval")

# Per-variant adapter cache + output directories. Selected at runtime
# from --variant. Defaults preserve the positive-only (``po``) behavior
# so existing run scripts / call sites stay byte-identical.
LOCAL_ADAPTER_CACHE_FOR: dict[str, Path] = {
    "po": Path("/workspace/adapters/i464_po"),
    "cn": Path("/workspace/adapters/i464_cn"),
    # #529 non-saturated-anchor cn re-run; adapter cache distinct from
    # the parent #464 cn cache so the two never collide on /workspace.
    "cn_i529": Path("/workspace/adapters/i529_cn"),
    # #533 lr=5e-6 corrective re-run; adapter cache DISTINCT from #529's
    # so the two never collide in the local pre-download cache.
    "cn_i533": Path("/workspace/adapters/i533_cn"),
    # #546 rank-reduction (r=16/alpha=32) corrective re-run of #533's
    # grid; cache distinct from #533's so the two never collide.
    "cn_i546": Path("/workspace/adapters/i546_cn"),
}
OUT_DIR_FOR: dict[str, Path] = {
    "po": Path("eval_results/issue_464/positive_only/cross_eval"),
    "cn": Path("eval_results/issue_464/contrastive_negatives/cross_eval"),
    "cn_i529": Path("eval_results/issue_529/contrastive_negatives/cross_eval"),
    "cn_i533": Path("eval_results/issue_533/contrastive_negatives/cross_eval"),
    "cn_i546": Path("eval_results/issue_546/contrastive_negatives/cross_eval"),
}
# HF Hub model-repo subpath PREFIX used to look up adapters per cell.
# Train writes positive-only adapters to ``adapters/i464_{arm}_seed{seed}_{persona}``
# and contrastive-negatives adapters to ``adapters/i464_{arm}_seed{seed}_cn_{persona}``
# (the train script inserts ``_cn`` as an INFIX before the persona, NOT a
# prefix — verified against the live HF upload ``i464_role_seed42_cn_pirate``);
# we mirror that distinction here so the two variants never overwrite
# each other's local cache or per-cell JSON.
ADAPTER_SUBPATH_FOR: dict[str, str] = {
    "po": "adapters/i464_{arm}_seed{seed}_{persona}",
    "cn": "adapters/i464_{arm}_seed{seed}_cn_{persona}",
    # #529: same cn infix as #464 but with an i529_ prefix AND an
    # ``_e{epoch}`` epoch suffix (the manipulated variable). Matches the
    # train script's hf_path_in_repo at args.issue=529.
    "cn_i529": "adapters/i529_{arm}_seed{seed}_cn_{persona}_e{epoch}",
    # #533 mirrors #529's shape with an i533_ prefix. Matches the train
    # script's hf_path_in_repo at args.issue=533.
    "cn_i533": "adapters/i533_{arm}_seed{seed}_cn_{persona}_e{epoch}",
    # #546 mirrors #533's shape with an i546_ prefix. Matches the train
    # script's hf_path_in_repo at args.issue=546.
    "cn_i546": "adapters/i546_{arm}_seed{seed}_cn_{persona}_e{epoch}",
}

# Legacy aliases (positive-only defaults) — kept for the smoke-test
# importers that referenced these constants before --variant existed.
LOCAL_ADAPTER_CACHE = LOCAL_ADAPTER_CACHE_FOR["po"]
OUT_DIR = OUT_DIR_FOR["po"]
PER_CELL_DIR = OUT_DIR / "per_cell"

# Per-variant seed sets. #529 deliberately bumps n=3 → 5 (plan §12 Assumption
# 8): the 3-seed CI in #464 was min/max of 3 points; 5 seeds gives a real
# paired bootstrap CI at the selected anchor.
SEEDS_FOR: dict[str, tuple[int, ...]] = {
    "po": (42, 137, 1337),
    "cn": (42, 137, 1337),
    "cn_i529": (42, 137, 1337, 7, 21),
    "cn_i533": (42, 137, 1337, 7, 21),
    "cn_i546": (42, 137, 1337, 7, 21),
}
# Legacy alias — kept for callers that referenced ``SEEDS`` before
# the per-variant split.
SEEDS = SEEDS_FOR["po"]
# Headline arms for the positive-only follow-up: system_plain, system_padded,
# role. role_nonsense / role_mismatch are NOT replicated here (they were
# diagnostic follow-ups to the parent's co-resident headline). The cn
# variant uses the SAME 3 arms.
PO_ARMS: tuple[enc.Arm, ...] = ("system_plain", "system_padded", "role")
# All probed eval encodings always carry the shared pirate marker ` ※`.
SHARED_MARKER_PERSONA: enc.Persona = "pirate"
# Epoch grid for the cn_i529 variant; see plan §4.1.
EPOCHS_I529: tuple[int, ...] = (1, 2, 3, 5)
# Epoch grid for the cn_i533 variant (lr=5e-6 corrective re-run); same
# as #529 — the grid IS the experimental dial and the single-variable
# contract holds it byte-stable.
EPOCHS_I533: tuple[int, ...] = (1, 2, 3, 5)
# Epoch grid for the cn_i546 variant (r=16/alpha=32 rank-reduction
# corrective re-run of #533); same grid — the single-variable contract
# with #533 holds it byte-stable.
EPOCHS_I546: tuple[int, ...] = (1, 2, 3, 5)


def _per_cell_dir_for(variant: str) -> Path:
    """Per-cell JSON directory for ``variant``."""
    if variant not in OUT_DIR_FOR:
        raise ValueError(f"unknown variant={variant!r}; want one of {list(OUT_DIR_FOR)}")
    return OUT_DIR_FOR[variant] / "per_cell"


def _all_po_cells(variant: str = "po") -> list[tuple[enc.Arm, int, enc.Persona, int | None]]:
    """Return all (arm, seed, persona, epoch) cells for the variant.

    For ``po`` / ``cn`` the epoch component is ``None`` (single anchor).
    For ``cn_i529`` / ``cn_i533`` / ``cn_i546`` it iterates the variant's
    epoch grid — the manipulated variable. The legacy 3-tuple ``po`` cell
    signature is preserved as ``(arm, seed, persona, None)``; downstream
    call sites unpack only the first three components when the variant is
    po/cn.
    """
    seeds = SEEDS_FOR.get(variant, SEEDS_FOR["po"])
    if variant == "cn_i529":
        return [
            (arm, seed, persona, epoch)
            for arm in PO_ARMS
            for seed in seeds
            for persona in enc.PERSONAS
            for epoch in EPOCHS_I529
        ]
    if variant == "cn_i533":
        return [
            (arm, seed, persona, epoch)
            for arm in PO_ARMS
            for seed in seeds
            for persona in enc.PERSONAS
            for epoch in EPOCHS_I533
        ]
    if variant == "cn_i546":
        return [
            (arm, seed, persona, epoch)
            for arm in PO_ARMS
            for seed in seeds
            for persona in enc.PERSONAS
            for epoch in EPOCHS_I546
        ]
    return [
        (arm, seed, persona, None) for arm in PO_ARMS for seed in seeds for persona in enc.PERSONAS
    ]


# Per-variant epoch grids (epoch-dimension variants only). Used by the
# cell-subset filters below and the --epoch validation in main().
_EPOCH_GRID_FOR: dict[str, tuple[int, ...]] = {
    "cn_i529": EPOCHS_I529,
    "cn_i533": EPOCHS_I533,
    "cn_i546": EPOCHS_I546,
}


def _apply_cell_filters(
    cells: list[tuple[enc.Arm, int, enc.Persona, int | None]],
    variant: str,
    *,
    arms: list[str] | None = None,
    seeds: list[int] | None = None,
    personas: list[str] | None = None,
    epochs: list[int] | None = None,
) -> list[tuple[enc.Arm, int, enc.Persona, int | None]]:
    """Restrict ``cells`` to a dispatcher-requested subset (smoke = sweep with one cell).

    Mirrors ``i546_cn_run.sh``'s ``ARMS/SEEDS/PERSONAS/EPOCHS_OVERRIDE``
    semantics so the eval phase honors the SAME cell subset the train loop
    ran (#546 round-3 fix for ``smoke-crosseval-enumerates-full-grid``: the
    overrides shaped only training, so a fresh-issue smoke enumerated the
    full grid here and 404'd on never-trained adapters).

    Each filter is optional; ``None`` = keep that dimension's full grid
    (the byte-stable default for po / cn / cn_i529 / cn_i533 production
    invocations). Values are validated against the variant's registered
    grid, and a non-empty result is required whenever any filter is
    active — a typo'd persona or out-of-grid epoch fails LOUD instead of
    silently evaluating zero cells. Raises ``ValueError`` on violations
    (main() surfaces it via ``ap.error``).
    """
    if epochs is not None and variant not in _EPOCH_GRID_FOR:
        raise ValueError(
            f"--epochs-filter only valid for epoch-grid variants "
            f"{sorted(_EPOCH_GRID_FOR)}; --variant {variant} has no epoch dimension"
        )
    spec: list[tuple[str, list | None, int, tuple]] = [
        ("--arms-filter", arms, 0, PO_ARMS),
        ("--seeds-filter", seeds, 1, SEEDS_FOR.get(variant, SEEDS_FOR["po"])),
        ("--personas-filter", personas, 2, enc.PERSONAS),
        ("--epochs-filter", epochs, 3, _EPOCH_GRID_FOR.get(variant, ())),
    ]
    out = cells
    any_active = False
    for flag, wanted, idx, allowed in spec:
        if wanted is None:
            continue
        any_active = True
        bad = sorted(set(wanted) - set(allowed))
        if bad:
            raise ValueError(f"{flag}: {bad} not in the {variant} grid {tuple(allowed)}")
        keep = set(wanted)
        out = [c for c in out if c[idx] in keep]
    if any_active and not out:
        raise ValueError(
            "cell-subset filters matched ZERO cells "
            f"(arms={arms} seeds={seeds} personas={personas} epochs={epochs}); "
            "refusing to run an empty eval"
        )
    return out


def _po_cell_label(arm: enc.Arm, seed: int, persona: enc.Persona, epoch: int | None = None) -> str:
    """Canonical cell label. For #529 includes the cn infix + epoch suffix.

    Shape:
      * po:   ``{arm}_seed{seed}_{persona}``
      * cn_i529: ``{arm}_seed{seed}_cn_{persona}_e{epoch}``

    The ``cn`` variant (#464) keeps the legacy po-shape filename (just
    arm/seed/persona — without the ``_cn_`` infix in the per-cell JSON
    name even though the HF adapter subpath carries it) to preserve
    backward compatibility with the existing #464 cn cross-eval output
    layout.
    """
    if epoch is not None:
        return f"{arm}_seed{seed}_cn_{persona}_e{epoch}"
    return f"{arm}_seed{seed}_{persona}"


def _parse_shard(spec: str | None) -> tuple[int, int]:
    """Parse ``--shard 'k-of-n'`` → (k, n). Mirrors parent eval."""
    if spec is None:
        return 0, 1
    s_idx, n = spec.split("-of-")
    s_idx_i = int(s_idx)
    n_i = int(n)
    if not (0 <= s_idx_i < n_i):
        raise ValueError(f"--shard {spec!r}: shard index {s_idx_i} not in [0, {n_i})")
    return s_idx_i, n_i


def _eval_encodings_for_cell(arm: enc.Arm, persona: enc.Persona) -> list[enc.EvalEncoding]:
    """Return the 3 eval encodings probed for ONE positive-only cell.

    Always: arm-family own + arm-family other + ``default_assistant``.

    The own/other encodings are in the SAME arm-family as the cell's
    training arm (a role-arm cell is NOT probed under a system encoding,
    and vice versa). This isolates the role-vs-system localization
    question the parent #464 sweep answered — but here under the
    co-residence-removed regime.
    """
    other_persona: enc.Persona = "villain" if persona == "pirate" else "pirate"
    if arm in ("system_plain", "system_padded"):
        own_enc: enc.EvalEncoding = f"system_{persona}"  # type: ignore[assignment]
        other_enc: enc.EvalEncoding = f"system_{other_persona}"  # type: ignore[assignment]
    elif arm == "role":
        own_enc = f"role_{persona}"  # type: ignore[assignment]
        other_enc = f"role_{other_persona}"  # type: ignore[assignment]
    else:
        raise ValueError(f"po follow-up only covers arms {PO_ARMS}; got arm={arm!r}")
    return [own_enc, other_enc, "default_assistant"]


def _load_R_canon_test() -> dict[str, dict[str, dict]]:
    """Load R_canon_test (HF fallback or local override). Mirrors phase4."""
    override_dir = os.environ.get("EPM_LOCAL_R_CANON_DIR")
    if override_dir:
        override_path = Path(override_dir) / "R_canon_test.json"
        if not override_path.exists():
            raise RuntimeError(
                f"EPM_LOCAL_R_CANON_DIR={override_dir!r} set but R_canon_test.json "
                f"missing at {override_path}."
            )
        logger.info("Using local R_canon override: %s", override_path)
        local = override_path
    else:
        local = LOCAL_DATA_DIR / "R_canon_test.json"
        if not local.exists():
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_R_PATH_PREFIX}/R_canon_test.json",
                revision="main",
            )
            import shutil

            shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    if payload.get("schema_version") != "i464_v2_matched_R":
        raise AssertionError(f"R_canon_test schema_version={payload.get('schema_version')!r}")
    return payload["completions"]


def _download_po_adapter(
    arm: enc.Arm,
    seed: int,
    persona: enc.Persona,
    variant: str = "po",
    epoch: int | None = None,
) -> str:
    """Per-file HF download for one cell's adapter; return its local dir.

    Mirrors ``i464_phase4_eval._download_adapter`` but on the variant's
    HF subpath:
      - ``po`` → ``adapters/i464_{arm}_seed{seed}_{persona}``
      - ``cn`` → ``adapters/i464_{arm}_seed{seed}_cn_{persona}``
      - ``cn_i529`` → ``adapters/i529_{arm}_seed{seed}_cn_{persona}_e{epoch}``
      - ``cn_i533`` → ``adapters/i533_{arm}_seed{seed}_cn_{persona}_e{epoch}``
      - ``cn_i546`` → ``adapters/i546_{arm}_seed{seed}_cn_{persona}_e{epoch}``
    (matches the train script's ``hf_path_in_repo`` for each variant).

    ``epoch`` is REQUIRED for the cn_i529 / cn_i533 / cn_i546 variants and
    IGNORED for the po / cn variants (their subpath templates don't reference it).

    Local-override env hook ``EPM_LOCAL_ADAPTER_OVERRIDE`` — when set,
    look for ``<override>/<variant-subpath>``; raise if missing (mirrors
    parent contract).
    """
    if variant not in ADAPTER_SUBPATH_FOR:
        raise ValueError(f"unknown variant={variant!r}; want one of {list(ADAPTER_SUBPATH_FOR)}")
    fmt_kwargs: dict[str, object] = {"arm": arm, "seed": seed, "persona": persona}
    if variant in ("cn_i529", "cn_i533", "cn_i546"):
        if epoch is None:
            raise ValueError(f"--variant {variant} requires epoch (None passed)")
        fmt_kwargs["epoch"] = epoch
    target_subpath = ADAPTER_SUBPATH_FOR[variant].format(**fmt_kwargs)
    cache_root = LOCAL_ADAPTER_CACHE_FOR[variant]
    override_root = os.environ.get("EPM_LOCAL_ADAPTER_OVERRIDE")
    if override_root:
        local_target = Path(override_root) / target_subpath
        if not (local_target / "adapter_model.safetensors").exists():
            raise RuntimeError(
                f"EPM_LOCAL_ADAPTER_OVERRIDE={override_root!r} set but adapter "
                f"missing at {local_target}/adapter_model.safetensors."
            )
        logger.info("Using local adapter override: %s", local_target)
        return str(local_target)

    from huggingface_hub import hf_hub_download

    cache_root.mkdir(parents=True, exist_ok=True)
    local_target = cache_root / target_subpath
    local_target.mkdir(parents=True, exist_ok=True)
    needed = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]
    for fname in needed:
        try:
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                revision="main",
                filename=f"{target_subpath}/{fname}",
                local_dir=cache_root,
            )
        except Exception as e:
            if fname in ("adapter_model.safetensors", "adapter_config.json"):
                raise RuntimeError(f"required {target_subpath}/{fname} not on HF: {e}") from e
            logger.debug("optional %s/%s missing: %s", target_subpath, fname, e)
    if not (local_target / "adapter_model.safetensors").exists():
        raise RuntimeError(f"adapter_model.safetensors missing at {local_target}.")
    return str(local_target)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the positive-only cross-eval."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--shard",
        default=None,
        help="Round-robin shard 'k-of-n' over the 18 cells (default: single shard).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip per-cell JSONs already written (re-use on crash recovery).",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len.",
    )
    ap.add_argument(
        "--smoke-n-q",
        type=int,
        default=0,
        help="If > 0, truncate Q_test to this many questions per probe (smoke).",
    )
    ap.add_argument(
        "--smoke-cells",
        nargs="+",
        default=None,
        help=(
            "If set, restrict to these cells (label form "
            "'system_plain_seed42_pirate'); smoke use only."
        ),
    )
    ap.add_argument(
        "--variant",
        choices=("po", "cn", "cn_i529", "cn_i533", "cn_i546"),
        default="po",
        help=(
            "Which follow-up's adapters to evaluate. ``po`` (default) = "
            "positive-only single-persona LoRAs at HF subpath "
            "``adapters/i464_{arm}_seed{seed}_{persona}``, outputs under "
            "``eval_results/issue_464/positive_only/cross_eval/``. ``cn`` = "
            "contrastive-negatives single-persona LoRAs at "
            "``adapters/i464_{arm}_seed{seed}_cn_{persona}``, outputs under "
            "``eval_results/issue_464/contrastive_negatives/cross_eval/``. "
            "``cn_i529`` = #529 non-saturated-anchor cn re-run; adapters at "
            "``adapters/i529_{arm}_seed{seed}_cn_{persona}_e{epoch}``, "
            "outputs under ``eval_results/issue_529/contrastive_negatives/"
            "cross_eval/``. Iterates 5 seeds x 4 epoch settings (1,2,3,5). "
            "``cn_i533`` = #533 lr=5e-6 corrective re-run of #529's grid; "
            "adapters at ``adapters/i533_{arm}_seed{seed}_cn_{persona}_e{epoch}``, "
            "outputs under ``eval_results/issue_533/contrastive_negatives/"
            "cross_eval/``. Same 5 seeds x 4 epochs grid as cn_i529. "
            "``cn_i546`` = #546 r=16/alpha=32 rank-reduction corrective "
            "re-run of #533's grid; adapters at "
            "``adapters/i546_{arm}_seed{seed}_cn_{persona}_e{epoch}``, "
            "outputs under ``eval_results/issue_546/contrastive_negatives/"
            "cross_eval/``. Same 5 seeds x 4 epochs grid as cn_i533."
        ),
    )
    ap.add_argument(
        "--epoch",
        type=int,
        default=None,
        help=(
            "Epoch grid filter for --variant cn_i529 / cn_i533 / cn_i546. When set, "
            "restrict evaluation to ONLY this epoch (1/2/3/5). Default "
            "None = iterate the variant's full epoch grid. IGNORED for "
            "--variant po/cn."
        ),
    )
    # Cell-subset filters — the dispatcher's smoke hooks (#546 round-3 fix
    # for `smoke-crosseval-enumerates-full-grid`). i546_cn_run.sh passes
    # these IFF the corresponding *_OVERRIDE env var is set, so the eval
    # phase enumerates the SAME cell subset the train loop ran. All four
    # default to None = full grid (byte-stable for every production
    # invocation, including cn_i529 / cn_i533).
    ap.add_argument(
        "--arms-filter",
        nargs="+",
        default=None,
        help=(
            "Restrict the cell grid to these arms (subset of "
            "system_plain/system_padded/role). Dispatcher smoke hook "
            "(i546_cn_run.sh ARMS_OVERRIDE); default = full grid."
        ),
    )
    ap.add_argument(
        "--seeds-filter",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Restrict the cell grid to these seeds (subset of the variant's "
            "registered seed set). Dispatcher smoke hook (SEEDS_OVERRIDE); "
            "default = full grid."
        ),
    )
    ap.add_argument(
        "--personas-filter",
        nargs="+",
        default=None,
        help=(
            "Restrict the cell grid to these training personas "
            "(pirate/villain). Dispatcher smoke hook (PERSONAS_OVERRIDE); "
            "default = full grid."
        ),
    )
    ap.add_argument(
        "--epochs-filter",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Restrict the cell grid to these epochs (subset of the variant's "
            "epoch grid; cn_i529 / cn_i533 / cn_i546 only). Dispatcher smoke "
            "hook (EPOCHS_OVERRIDE); default = full grid."
        ),
    )
    args = ap.parse_args(argv)

    shard_idx, n_shards = _parse_shard(args.shard)
    out_dir = OUT_DIR_FOR[args.variant]
    per_cell_dir = _per_cell_dir_for(args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "variant=%s out_dir=%s per_cell_dir=%s",
        args.variant,
        out_dir,
        per_cell_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)
    # Defense-in-depth: even though enc.assert_token_ids already covers it,
    # re-assert the shared marker id here (a future encodings refactor
    # could change the constant without changing the assert).
    shared_ids = tokenizer.encode(enc.MARKER_PIRATE_TEXT, add_special_tokens=False)
    if shared_ids != [enc.MARKER_PIRATE_ID]:
        raise AssertionError(
            f"shared marker {enc.MARKER_PIRATE_TEXT!r} tokenizes to {shared_ids}, "
            f"expected [{enc.MARKER_PIRATE_ID}]"
        )

    q_test = load_q_test_extended_50()
    R_canon_test = _load_R_canon_test()

    if args.smoke_n_q > 0:
        q_test = q_test[: args.smoke_n_q]
        logger.warning("SMOKE: truncated Q_test to %d questions", len(q_test))

    # Preflight: verify R_canon_test covers EVERY (persona, q_test) the
    # downstream eval loop will consume BEFORE we download adapters or
    # spin up vLLM. The eval loop (``_build_probes_for_eval_marker``,
    # imported from phase4) indexes
    # ``R_canon_test[R_persona][q]["response_text"]`` with no in-loop
    # guard; a subset / drifted artifact would crash mid-eval with a
    # bare ``KeyError`` AFTER vLLM is up, wasting GPU spend on every
    # cn_i529 cell launch. Round-3 closure for
    # ``eval-po-r-canon-coverage-unverified``: the same gate the parent
    # eval uses, called on the actual production cn_i529 entrypoint.
    # All probed eval encodings always carry the shared pirate marker
    # ``enc.MARKER_PIRATE_TEXT`` and R splices via
    # ``enc.persona_for_eval_encoding(e_eval)``, which maps to one of
    # ``enc.PERSONAS`` (pirate / villain) for system_/role_ encodings
    # and to the bare default for ``default_assistant`` — the eval loop
    # ONLY indexes R_canon_test with personas in ``enc.PERSONAS``, so
    # that's the required-coverage set.
    assert_r_canon_test_coverage(R_canon_test, q_test, enc.PERSONAS)
    logger.info(
        "R_canon_test coverage: %d personas x %d q_test rows all present.",
        len(set(enc.PERSONAS)),
        len(q_test),
    )

    all_cells = _all_po_cells(variant=args.variant)
    if args.variant in ("cn_i529", "cn_i533", "cn_i546") and args.epoch is not None:
        variant_epochs = _EPOCH_GRID_FOR[args.variant]
        if args.epoch not in variant_epochs:
            ap.error(f"--epoch {args.epoch} not in {args.variant} grid={variant_epochs}")
        all_cells = [c for c in all_cells if c[3] == args.epoch]
        logger.info(
            "variant=%s --epoch=%d filter: %d cells",
            args.variant,
            args.epoch,
            len(all_cells),
        )
    cell_filters_active = any(
        f is not None
        for f in (args.arms_filter, args.seeds_filter, args.personas_filter, args.epochs_filter)
    )
    if cell_filters_active:
        try:
            all_cells = _apply_cell_filters(
                all_cells,
                args.variant,
                arms=args.arms_filter,
                seeds=args.seeds_filter,
                personas=args.personas_filter,
                epochs=args.epochs_filter,
            )
        except ValueError as e:
            ap.error(str(e))
        logger.info(
            "cell-subset filters active (arms=%s seeds=%s personas=%s epochs=%s): %d cells",
            args.arms_filter,
            args.seeds_filter,
            args.personas_filter,
            args.epochs_filter,
            len(all_cells),
        )
    if args.smoke_cells:
        wanted = set(args.smoke_cells)
        all_cells = [c for c in all_cells if _po_cell_label(c[0], c[1], c[2], c[3]) in wanted]
        logger.warning("SMOKE: restricted to %d cell(s)", len(all_cells))

    my_cells = [c for k, c in enumerate(all_cells) if k % n_shards == shard_idx]
    logger.info(
        "Shard %d/%d owns %d cells: %s",
        shard_idx,
        n_shards,
        len(my_cells),
        [_po_cell_label(c[0], c[1], c[2], c[3]) for c in my_cells],
    )

    adapter_paths: dict[tuple[enc.Arm, int, enc.Persona, int | None], str] = {
        (a, s, p, e): _download_po_adapter(a, s, p, variant=args.variant, epoch=e)
        for (a, s, p, e) in my_cells
    }

    # vLLM late import; one engine for all cells, LoRARequest hot-swap.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_seq_len,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    # Base log-prob cache, keyed on e_eval (marker is ALWAYS the shared
    # pirate marker, so the second cache-key dimension is constant).
    base_cache: dict[enc.EvalEncoding, dict] = {}

    def _get_base(e_eval: enc.EvalEncoding) -> dict:
        if e_eval in base_cache:
            return base_cache[e_eval]
        prompts, slots = _build_probes_for_eval_marker(
            e_eval, SHARED_MARKER_PERSONA, tokenizer, q_test, R_canon_test
        )
        marker_id = enc.marker_id_for(SHARED_MARKER_PERSONA)
        t0 = time.time()
        outs = llm.generate(prompts, sp, lora_request=None)
        b_logps, b_argmax = _extract_marker_logp(
            outs, slots, marker_id, cell_label=f"BASE/{e_eval}/marker_{SHARED_MARKER_PERSONA}"
        )
        logger.info(
            "BASE e_eval=%s marker=%s done in %.1fs (logp_mean=%.2f argmax=%.2f)",
            e_eval,
            SHARED_MARKER_PERSONA,
            time.time() - t0,
            float(np.mean(b_logps)),
            sum(b_argmax) / len(b_argmax),
        )
        base_cache[e_eval] = {
            "prompts": prompts,
            "slots": slots,
            "b_logps": b_logps,
            "b_argmax": b_argmax,
            "marker_id": marker_id,
        }
        return base_cache[e_eval]

    for arm, seed, persona, epoch in my_cells:
        cell_label = _po_cell_label(arm, seed, persona, epoch)
        lora_req = LoRARequest(
            lora_name=cell_label,
            lora_int_id=all_cells.index((arm, seed, persona, epoch)) + 1,
            lora_path=adapter_paths[(arm, seed, persona, epoch)],
        )
        for e_eval in _eval_encodings_for_cell(arm, persona):
            out_path = per_cell_dir / f"{cell_label}__{e_eval}.json"
            if args.resume and out_path.exists() and out_path.stat().st_size > 0:
                continue
            base = _get_base(e_eval)
            t0 = time.time()
            outs = llm.generate(base["prompts"], sp, lora_request=lora_req)
            t_logps, t_argmax = _extract_marker_logp(
                outs,
                base["slots"],
                base["marker_id"],
                cell_label=f"TRAINED/{cell_label}/{e_eval}/marker_{SHARED_MARKER_PERSONA}",
            )
            t_arr = np.array(t_logps, dtype=float)
            b_arr = np.array(base["b_logps"], dtype=float)
            delta = t_arr - b_arr
            payload = {
                "cell": cell_label,
                "arm": arm,
                "seed": seed,
                "training_persona": persona,
                "marker_persona": SHARED_MARKER_PERSONA,
                "e_eval": e_eval,
                "marker_id": base["marker_id"],
                "n_probes": len(t_logps),
                "g_logprob": float(t_arr.mean()),
                "b_logprob": float(b_arr.mean()),
                "delta_g": float(delta.mean()),
                "emission_recompute_rate": sum(t_argmax) / len(t_argmax),
                "logp_floor": LOGP_FLOOR,
                "g_logps_per_q": t_logps,
                "b_logps_per_q": list(base["b_logps"]),
                "g_argmax_marker_per_q": t_argmax,
                "b_argmax_marker_per_q": list(base["b_argmax"]),
            }
            # cn_i529 / cn_i533: record the epoch so anchor-selection /
            # analyze can find E* in O(1) without parsing it back out of
            # the label.
            if epoch is not None:
                payload["epoch"] = epoch
                payload["variant"] = args.variant
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(out_path)
            logger.info(
                "cell=%s e_eval=%s g=%.3f b=%.3f Δ=%+.3f emit=%.3f in %.1fs -> %s",
                cell_label,
                e_eval,
                payload["g_logprob"],
                payload["b_logprob"],
                payload["delta_g"],
                payload["emission_recompute_rate"],
                time.time() - t0,
                out_path,
            )


if __name__ == "__main__":
    main()
