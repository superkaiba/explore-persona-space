#!/usr/bin/env python3
"""Issue #1415 — Phase-1 GPU driver (plan v5 deliverable 3).

Phases (all single-GPU A100-80 in production; ``--tiny`` runs the FULL control
flow on CPU with a from-config 2-layer Qwen model):

- **1b** unhooked generation under c (baseline arm ``hf_nohook_base``) and
  under c' (context-swap ceiling arm ``ctx_swap_ceil``): pairs x 2 x N draws.
- **1a** capture V_c (prefix AND context arms) + V_a over the 1b completions
  at the sweep layers.
- **1c** Delta-addition generation (Delta = V_c(c') - V_c(c), per extraction
  arm): full alpha grid at the primary layer, then the remaining layers at the
  coherence-selected alpha (largest grid alpha whose condition passes the
  >=50%-coherent gate; on total failure the gate lowers alpha by x0.5 — one
  sub-grid retry — and records the operating alpha per pair), then the
  all-positions variant at the primary layer + selected alpha.
- **1d** r_B arm: ``issue779_monitoring/r_b/{trait}.pt`` (shape asserted
  (28, 3584) in production), alpha search on a 5-pair subset, then the full
  pair set x traits x N at the selected alpha.
- **1e** teacher-forced V_a capture over EVERY persisted steered completion
  cell (1c grid/retry/layers/allpos + 1d full) at the sweep layers — the plan
  §4.10 DV (a) input (round-2 fix for the missing H1 headline-DV leg). Also
  writes the map-transport canonical files
  ``activations_steered/<pair_id>__<arm>.pt`` (the selected operating-alpha
  cell at the primary layer). NOT a plan §9 row — the compute addition
  (~7,000 teacher-forced samples ≈ 0.5-0.75 GPU-h at the 1a basis) is
  recorded in the round-2 implementation report.
- ``--pilot``: 1 pair x alpha=1.0 x primary layer, replicated to
  B = ``--gen-batch`` identical rows (the SWEEP's chunk shape — never batch-1,
  which over-reads s/sample by ~B on bandwidth-bound HF decode; crash-fix for
  att-20260716-160022) x N draws (+ the all-positions variant); measures
  s/sample at the sweep shape, logs + persists it, and the full sweep HALTS by
  design when s/sample > 4.7 unless ``--force`` (``pilot_gate_report.json``,
  exit ``RC_PILOT_GATE=7`` — artifact-routed in the dispatcher like K1/K2).

Pre-registered in-run kill criteria (plan v5 §3; round-2 fix):

- **K1** (post-1a): the context-swap ceiling must show real answer-side
  separation. Geometric half IN-DRIVER: per pair, the split-half (even/odd
  draws) answer-target direction cosine, max over the sweep layers, vs the
  p97.5 of a random-direction max-over-layers band; >80% of pairs within the
  band -> ABORT the 1c/1d sweep, ``k1_report.json``, exit ``RC_K1_ABORT=4``.
  The judge half (ceiling judge-shift < 5 pts) runs OFF-pod and is a
  DEFERRED-to-analysis warning in ``issue1415_judge.py`` (recorded deviation:
  the in-driver abort keys on the geometric half alone).
- **K2** (inside 1c): the primary-layer alpha grid runs the pilot pair + the
  FIRST 5 pairs first; if the coherence gate fails at ALL grid alphas for
  >50% of those units -> HALT 1c/1d, ``k2_report.json``, exit ``RC_K2_HALT=5``.

Under ``--tiny`` the kill-criteria VERDICTS are computed + persisted but the
aborts are demoted to loud log lines (gotchas.md: smoke/production gate
calibration — a random-weight 2-layer model makes the production-calibrated
verdicts meaningless at smoke scale); the abort branches are unit-pinned in
``tests/test_issue1415_phase1_driver.py``. ``--ignore-kill-criteria`` demotes
them in production too (explicit override, always recorded in the reports).

Checkpoint-per-cell: ``phase1_manifest.json`` marks completed cells (keyed on
EVERY output-affecting regime knob); a rerun skips them. Artifacts write
incrementally: completions -> HF data-repo prefix ``raw_completions/issue_1415/``,
capture tensors -> ``analysis_tensors/issue_1415/activations/`` (+ steered
captures -> ``analysis_tensors/issue_1415/activations_steered/``), per-cell
metadata JSON -> the (git) ``--out-root`` (default ``eval_results/issue_1415/phase1``).
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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import torch  # noqa: E402

from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    DeltaHook,
    capture_vectors,
    coherence_check,
    condition_passes,
    generate_batch,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1415_pair_bank  # noqa: E402  (self-build of the pair bank on fresh instances)

logger = logging.getLogger("issue1415.phase1")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
I779_PREFIX = "issue779_monitoring"
TRAITS = ("evil", "sycophancy", "hallucination")

LAYERS_FULL = (7, 10, 14, 17, 20, 21, 24)
PRIMARY_LAYER_FULL = 20
ALPHA_GRID = (0.5, 1.0, 2.0, 4.0)
N_DRAWS_FULL = 10
MAX_NEW_TOKENS_FULL = 1024
SEED_BASE = 42
TEMPERATURE = 1.0
HIDDEN_FULL = 3584
N_MODEL_LAYERS_FULL = 28
PILOT_MAX_S_PER_SAMPLE = 4.7  # plan v5 §9 pilot gate

EXTRACTION_ARMS = ("prefix", "context")  # Delta from v_c_prefix vs v_c_context

RAW_PREFIX = "raw_completions/issue_1415"
TENSOR_PREFIX = "analysis_tensors/issue_1415/activations"
STEERED_TENSOR_PREFIX = "analysis_tensors/issue_1415/activations_steered"

# Steered-cell phases whose completions get a phase-1e V_a capture.
CAPTURED_PHASES = (
    "phase1c_grid",
    "phase1c_retry",
    "phase1c_layers",
    "phase1c_allpos",
    "phase1d_full",
)

# Kill-criteria constants (plan v5 §3). Distinct exit codes so the dispatcher
# can route the domain HALT on the artifact (gotchas.md: wrap-script
# route-on-artifact).
RC_K1_ABORT = 4
RC_K2_HALT = 5
RC_PILOT_GATE = 7  # designed pilot-gate HALT (plan §9 timing gate; §13 descope ladder)
K1_NO_SEP_FRAC = 0.8  # fire when > 80% of pairs show no separation
K1_NULL_DRAWS = 500
K1_SEED = 1415
K2_FIRST_PAIRS = 5
K2_FAIL_FRAC = 0.5  # fire when > 50% of pilot+first-5 units fail at ALL alphas


# ── config ────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Resolved run configuration (every field here is an output-affecting knob
    or an IO root; the output-affecting subset enters the manifest regime)."""

    tiny: bool
    pilot_only: bool
    force: bool
    out_root: Path
    bulk_root: Path
    pair_bank_path: Path
    upload_mode: str  # "hf" | "local-mirror"
    model_id: str
    n_draws: int
    max_new_tokens: int
    gen_batch: int
    capture_batch: int
    layers: tuple[int, ...]
    primary_layer: int
    alpha_grid: tuple[float, ...]
    seed_base: int
    temperature: float
    hidden: int
    n_model_layers: int
    tiny_pairs: int
    device: str
    # Kill-criteria enforcement (NOT part of the manifest regime: K1/K2 only
    # truncate the run, they never change any cell's output).
    enforce_kill_criteria: bool = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Issue #1415 phase-1 driver (1b/1a/1c/1d/1e + pilot + K1/K2 kill criteria)."
    )
    ap.add_argument("--tiny", action="store_true", help="from-config 2-layer CPU model (smoke)")
    ap.add_argument("--pilot", action="store_true", dest="pilot_only", help="pilot timing only")
    ap.add_argument("--force", action="store_true", help="override the 4.7 s/sample pilot gate")
    ap.add_argument("--out-root", type=Path, default=None, help="metadata/manifest root (git)")
    ap.add_argument("--bulk-root", type=Path, default=None, help="completions/tensors staging")
    ap.add_argument("--pair-bank", type=Path, default=None)
    ap.add_argument("--upload", choices=("hf", "local-mirror"), default=None)
    ap.add_argument("--n-draws", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--gen-batch", type=int, default=8, help="contexts per batched generate call")
    ap.add_argument("--capture-batch", type=int, default=8)
    ap.add_argument("--tiny-pairs", type=int, default=2, help="pairs in the --tiny synthetic bank")
    ap.add_argument(
        "--ignore-kill-criteria",
        action="store_true",
        help="compute + persist the K1/K2 verdicts but demote the aborts to log lines",
    )
    return ap.parse_args(argv)


def build_config(args: argparse.Namespace) -> RunConfig:
    """Resolve mode-dependent defaults (tiny smoke roots NEVER collide with the
    canonical eval_results paths — smoke outputs must not overwrite committed
    artifacts)."""
    if args.tiny:
        out_root = args.out_root or REPO_ROOT / "data" / "issue_1415" / "tiny_smoke" / "out"
        bulk_root = args.bulk_root or REPO_ROOT / "data" / "issue_1415" / "tiny_smoke" / "bulk"
        pair_bank = args.pair_bank or out_root.parent / "pair_bank_tiny.json"
        return RunConfig(
            tiny=True,
            pilot_only=args.pilot_only,
            force=args.force,
            out_root=out_root,
            bulk_root=bulk_root,
            pair_bank_path=pair_bank,
            upload_mode=args.upload or "local-mirror",
            model_id=MODEL_ID,  # real tokenizer + config family, tiny dims
            n_draws=args.n_draws or 2,
            max_new_tokens=args.max_new_tokens or 16,
            gen_batch=args.gen_batch,
            capture_batch=args.capture_batch,
            layers=(0, 1),
            primary_layer=1,
            alpha_grid=ALPHA_GRID,
            seed_base=SEED_BASE,
            temperature=TEMPERATURE,
            hidden=64,
            n_model_layers=2,
            tiny_pairs=args.tiny_pairs,
            device="cpu",
            # gate-calibration parity (gotchas.md): tiny computes + persists the
            # K1/K2 verdicts but never aborts on them (random-weight model).
            enforce_kill_criteria=False,
        )
    return RunConfig(
        tiny=False,
        pilot_only=args.pilot_only,
        force=args.force,
        out_root=args.out_root or REPO_ROOT / "eval_results" / "issue_1415" / "phase1",
        bulk_root=args.bulk_root or REPO_ROOT / "data" / "issue_1415" / "phase1",
        pair_bank_path=args.pair_bank or REPO_ROOT / "data" / "issue_1415" / "pair_bank.json",
        upload_mode=args.upload or "hf",
        model_id=MODEL_ID,
        n_draws=args.n_draws or N_DRAWS_FULL,
        max_new_tokens=args.max_new_tokens or MAX_NEW_TOKENS_FULL,
        gen_batch=args.gen_batch,
        capture_batch=args.capture_batch,
        layers=LAYERS_FULL,
        primary_layer=PRIMARY_LAYER_FULL,
        alpha_grid=ALPHA_GRID,
        seed_base=SEED_BASE,
        temperature=TEMPERATURE,
        hidden=HIDDEN_FULL,
        n_model_layers=N_MODEL_LAYERS_FULL,
        tiny_pairs=args.tiny_pairs,
        device="cuda",
        enforce_kill_criteria=not args.ignore_kill_criteria,
    )


# ── small IO helpers ──────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj) -> None:
    """Atomic JSON write (tmp + os.replace) — checkpoint-per-cell safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _save_pt_atomic(path: Path, obj) -> None:
    """Atomic torch.save (tmp + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _fmt(alpha: float) -> str:
    """Deterministic alpha token for cell ids/paths (1.0 -> '1', 0.5 -> '0.5')."""
    return f"{alpha:g}"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_REPRO_CACHE: dict | None = None


def _repro(cfg: RunConfig) -> dict:
    """Reproducibility metadata for every persisted result (git commit, env
    versions, timestamp)."""
    global _REPRO_CACHE
    if _REPRO_CACHE is None:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                env={**os.environ},
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:  # metadata-only fallback
            logger.warning("git rev-parse failed (%s) — recording commit=unknown", exc)
            commit = "unknown"
        import transformers

        _REPRO_CACHE = {
            "git_commit": commit,
            # str(): torch.__version__ is a TorchVersion object — a non-str value
            # here breaks weights_only=True reload of the capture .pt blobs.
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
        }
    return {
        **_REPRO_CACHE,
        "model_id": cfg.model_id,
        "tiny": cfg.tiny,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── manifest (checkpoint-per-cell + resume) ───────────────────────────


class Manifest:
    """Completed-cell registry. Resume is keyed on EVERY output-affecting
    regime knob — a mismatched regime FAILS LOUD instead of silently reusing
    wrong cached cells (#722 r3)."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data

    @classmethod
    def load_or_init(cls, path: Path, regime: dict) -> Manifest:
        if path.exists():
            data = json.loads(path.read_text())
            if data.get("regime") != regime:
                raise RuntimeError(
                    f"manifest regime mismatch at {path}:\n"
                    f"  existing: {json.dumps(data.get('regime'), sort_keys=True)}\n"
                    f"  current:  {json.dumps(regime, sort_keys=True)}\n"
                    "resume must not cross regimes — use a fresh --out-root/--bulk-root"
                )
        else:
            data = {"regime": regime, "cells": {}}
            _write_json_atomic(path, data)
        return cls(path, data)

    def done(self, cell_id: str) -> bool:
        return cell_id in self.data["cells"]

    def get(self, cell_id: str) -> dict | None:
        return self.data["cells"].get(cell_id)

    def mark(self, cell_id: str, info: dict | None = None) -> None:
        self.data["cells"][cell_id] = {
            "completed_at": datetime.now(UTC).isoformat(),
            **(info or {}),
        }
        _write_json_atomic(self.path, self.data)


def _regime(cfg: RunConfig, pair_bank_sha: str) -> dict:
    return {
        "model_id": cfg.model_id,
        "tiny": cfg.tiny,
        "n_draws": cfg.n_draws,
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
        "seed_base": cfg.seed_base,
        "gen_batch": cfg.gen_batch,  # chunk composition affects sampled draws
        "layers": list(cfg.layers),
        "primary_layer": cfg.primary_layer,
        "alpha_grid": list(cfg.alpha_grid),
        "hidden": cfg.hidden,
        "n_model_layers": cfg.n_model_layers,
        "pair_bank_sha256": pair_bank_sha,
    }


# ── pair bank ─────────────────────────────────────────────────────────

_TINY_QUESTIONS = (
    "What is the best way to learn?",
    "How do airplanes stay in the air?",
    "Why is the sky blue?",
)


def _tiny_bank(n_pairs: int) -> dict:
    """Synthetic pair bank in the exact pair_bank.json schema (tiny smoke)."""
    assert 1 <= n_pairs <= len(_TINY_QUESTIONS), n_pairs
    pairs = []
    for i in range(n_pairs):
        q = _TINY_QUESTIONS[i % len(_TINY_QUESTIONS)]
        pairs.append(
            {
                "pair_id": f"tiny_{i:02d}",
                "pair_type": "matched" if i % 2 == 0 else "cross",
                "ctx_c": {"system": None, "user": q},
                "ctx_cprime": {"system": "You are a pirate captain.", "user": q},
                # a #685 behavior label so issue1415_judge.resolve_rubric resolves
                # OFFLINE in the tiny smoke (a fabricated label like "pirate" has
                # no rubric source and crashes the judge phase — round-C smoke fix)
                "trait_or_behavior": "hedging",
            }
        )
    return {"metadata": {"issue": 1415, "tiny": True}, "pairs": pairs}


def load_pairs(cfg: RunConfig) -> list[dict]:
    """Load (self-building on a fresh instance — data/ is gitignored) and
    schema-validate the pair bank."""
    path = cfg.pair_bank_path
    if not path.exists():
        if cfg.tiny:
            _write_json_atomic(path, _tiny_bank(cfg.tiny_pairs))
        else:
            logger.info("[phase=setup] pair bank missing at %s — self-building", path)
            issue1415_pair_bank.build_pair_bank(path)
    bank = json.loads(path.read_text())
    pairs = bank["pairs"]
    if not cfg.tiny:
        assert len(pairs) == 28, f"expected the 28-pair bank, got {len(pairs)}"
    for p in pairs:
        assert {"pair_id", "pair_type", "ctx_c", "ctx_cprime", "trait_or_behavior"} <= set(p), (
            sorted(p)
        )
    assert len({p["pair_id"] for p in pairs}) == len(pairs)
    return pairs


# ── model ─────────────────────────────────────────────────────────────


def load_model_and_tokenizer(cfg: RunConfig):
    """Production: bf16 Qwen-2.5-7B-Instruct pinned to cuda:0 (never
    device_map='auto' — silent CPU offload, gotchas). Tiny: from-config
    2-layer same-arch model on CPU (the committed unit-test pattern)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)  # loaded ONCE (429 gotcha)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if cfg.tiny:
        mcfg = AutoConfig.from_pretrained(cfg.model_id)
        mcfg.hidden_size = cfg.hidden
        mcfg.intermediate_size = 2 * cfg.hidden
        mcfg.num_hidden_layers = cfg.n_model_layers
        mcfg.num_attention_heads = 4
        mcfg.num_key_value_heads = 2
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(mcfg).to(torch.float32)
    else:
        assert torch.cuda.is_available(), "full phase-1 requires CUDA (use --tiny for CPU smoke)"
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16)
        model = model.to("cuda:0")
    assert model.config.hidden_size == cfg.hidden, (model.config.hidden_size, cfg.hidden)
    assert model.config.num_hidden_layers == cfg.n_model_layers
    model.eval()
    return model, tok


# ── delta source (pair Deltas from 1a captures; r_B from #779) ────────


def _unwrap_rb_tensor(obj) -> torch.Tensor:
    """Accept a raw (L, H) tensor or a dict carrying exactly one such tensor;
    anything else fails loud with the observed structure."""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for key in ("r_b", "rb", "direction", "vector"):
            if isinstance(obj.get(key), torch.Tensor):
                return obj[key]
        two_d = {k: v for k, v in obj.items() if isinstance(v, torch.Tensor) and v.dim() == 2}
        if len(two_d) == 1:
            return next(iter(two_d.values()))
    raise RuntimeError(
        f"unrecognized r_B blob: type={type(obj).__name__} "
        f"keys={sorted(obj) if isinstance(obj, dict) else None}"
    )


class DeltaSource:
    """Resolves steering vectors: pair Deltas (from the phase-1a capture .pt,
    per extraction arm + layer) and #779 r_B trait rows."""

    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self._pair_cache: dict[str, dict] = {}
        self._rb_cache: dict[str, torch.Tensor] = {}

    def resolve(self, key: tuple, layer: int) -> torch.Tensor:
        if key[0] == "pair":
            _, pair_id, arm = key
            return self.pair_delta(pair_id, arm, layer)
        if key[0] == "rb":
            _, trait = key
            return self.rb_delta(trait, layer)
        raise ValueError(key)

    def pair_delta(self, pair_id: str, arm: str, layer: int) -> torch.Tensor:
        assert arm in EXTRACTION_ARMS, arm
        if pair_id not in self._pair_cache:
            path = self.cfg.bulk_root / "activations" / f"{pair_id}.pt"
            assert path.exists(), f"phase-1a capture missing for {pair_id}: {path}"
            self._pair_cache[pair_id] = torch.load(path, map_location="cpu", weights_only=True)
        rec = self._pair_cache[pair_id]
        idx = list(self.cfg.layers).index(layer)  # captures store only the sweep layers
        d = rec["cprime"][f"v_c_{arm}"][idx] - rec["c"][f"v_c_{arm}"][idx]
        assert d.shape == (self.cfg.hidden,), d.shape
        return d

    def preload_rb(self) -> None:
        """Stage + shape-assert every r_B input up-front (fail EARLY, before
        any model load / GPU spend)."""
        for trait in TRAITS:
            self.rb_delta(trait, self.cfg.primary_layer)

    def rb_delta(self, trait: str, layer: int) -> torch.Tensor:
        assert trait in TRAITS, trait
        if trait not in self._rb_cache:
            if self.cfg.tiny:
                torch.manual_seed(1415 + TRAITS.index(trait))
                rb = torch.randn(self.cfg.n_model_layers, self.cfg.hidden)
            else:
                from huggingface_hub import hf_hub_download

                local = Path(
                    hf_hub_download(
                        HF_DATA_REPO, f"{I779_PREFIX}/r_b/{trait}.pt", repo_type="dataset"
                    )
                )
                rb = _unwrap_rb_tensor(torch.load(local, map_location="cpu", weights_only=True))
            # plan v5 1d contract: (28, 3584) in production (config-scaled in tiny)
            assert rb.shape == (self.cfg.n_model_layers, self.cfg.hidden), (trait, rb.shape)
            self._rb_cache[trait] = rb.float()
        return self._rb_cache[trait][layer]  # r_B rows index MODEL layers directly


# ── upload boundary ───────────────────────────────────────────────────


def _hf_upload(local: Path, path_in_repo: str) -> None:
    """One folder/file commit to the HF data repo via the canonical hub helper.

    ``hub._upload`` fail-softs to "" when its inner retry budget exhausts, so
    a no-path return gets a bounded OUTER retry then a fail-loud raise
    (upload-policy § fail-fast outer retry, #1315).
    """
    from explore_persona_space.orchestrate import hub

    delays = (30.0, 60.0, 120.0)
    for attempt in range(len(delays) + 1):
        url = hub._upload(
            local,
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            upload_as_file=local.is_file(),
        )
        if url:
            logger.info("[upload] %s -> %s", local, url)
            return
        if attempt < len(delays):
            logger.warning(
                "[upload] no path returned for %s (attempt %d) — retrying in %.0fs",
                path_in_repo,
                attempt + 1,
                delays[attempt],
            )
            time.sleep(delays[attempt])
    raise RuntimeError(f"upload returned no path for {path_in_repo} after {len(delays) + 1} tries")


def upload_artifact(cfg: RunConfig, local: Path, path_in_repo: str) -> None:
    """The single upload boundary (mockable). ``local-mirror`` (tiny default)
    copies into ``bulk_root/hf_mirror/<path_in_repo>`` through the identical
    call path so the control flow never forks."""
    if cfg.upload_mode == "local-mirror":
        dest = cfg.bulk_root / "hf_mirror" / path_in_repo
        if local.is_dir():
            shutil.copytree(local, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dest)
        logger.info("[upload] mirrored %s -> %s", local, dest)
        return
    assert cfg.upload_mode == "hf", cfg.upload_mode
    _hf_upload(local, path_in_repo)


def _upload_phase(
    cfg: RunConfig, state: Manifest, summary: dict, local_rel: str, remote: str
) -> None:
    """Incremental per-phase upload (ONE folder commit per phase — never a
    per-file loop; 256-commits/hr + per-file-504 gotchas). Re-uploads when the
    staged file count changed since the recorded upload."""
    local = cfg.bulk_root / local_rel
    if not local.exists():
        logger.info("[upload] nothing staged under %s — skipping", local_rel)
        return
    n_items = sum(1 for f in local.rglob("*") if f.is_file())
    marker = f"upload/{local_rel}"
    prior = state.get(marker)
    if prior and prior.get("n_items") == n_items:
        summary["uploads_skipped"] += 1
        return
    upload_artifact(cfg, local, remote)
    state.mark(marker, {"n_items": n_items, "remote": remote})
    summary["uploads"] += 1


# ── generation cells ──────────────────────────────────────────────────


@dataclass
class GenCell:
    cell_id: str
    phase: str
    pair_id: str
    context: dict
    layer: int | None = None
    alpha: float | None = None
    all_positions: bool = False
    delta_key: tuple | None = None  # ("pair", pair_id, arm) | ("rb", trait) | None
    extra: dict = field(default_factory=dict)


def _cell_meta_path(cfg: RunConfig, cell_id: str) -> Path:
    return cfg.out_root / "cells" / f"{cell_id}.json"


def load_cell_meta(cfg: RunConfig, cell_id: str) -> dict:
    p = _cell_meta_path(cfg, cell_id)
    assert p.exists(), f"manifest marks {cell_id} complete but metadata missing: {p}"
    return json.loads(p.read_text())


def _load_draws(cfg: RunConfig, cell_id: str) -> list[str]:
    p = cfg.bulk_root / "raw_completions" / f"{cell_id}.json"
    assert p.exists(), f"completions for completed cell {cell_id} missing: {p} (bulk root wiped?)"
    return json.loads(p.read_text())["draws"]


def _persist_gen_cell(
    cfg: RunConfig,
    state: Manifest,
    cell: GenCell,
    draws: list[str],
    s_per_sample: float,
    chunk_members: list[str],
) -> None:
    """Per-cell incremental persistence: completions (bulk, HF-bound),
    metadata (git-bound), manifest mark — the moment the cell completes."""
    assert len(draws) == cfg.n_draws, (cell.cell_id, len(draws))
    flags = coherence_check(draws)
    comp_rel = f"raw_completions/{cell.cell_id}.json"
    common = {
        "cell_id": cell.cell_id,
        "phase": cell.phase,
        "pair_id": cell.pair_id,
        "context": cell.context,
        "layer": cell.layer,
        "alpha": cell.alpha,
        "all_positions": cell.all_positions,
        "delta_key": list(cell.delta_key) if cell.delta_key else None,
        "n_draws": cfg.n_draws,
        "seed_base": cfg.seed_base,
        "temperature": cfg.temperature,
        "max_new_tokens": cfg.max_new_tokens,
        **cell.extra,
    }
    _write_json_atomic(cfg.bulk_root / comp_rel, {**common, "draws": draws, "repro": _repro(cfg)})
    _write_json_atomic(
        _cell_meta_path(cfg, cell.cell_id),
        {
            **common,
            "coherence_flags": flags,
            "n_coherent": sum(flags),
            "passes_gate": condition_passes(flags),
            "gen_s_per_sample": s_per_sample,
            "chunk_members": chunk_members,
            "completions_file": comp_rel,
            "repro": _repro(cfg),
        },
    )
    state.mark(cell.cell_id, {"phase": cell.phase})


def run_gen_cells(
    cfg: RunConfig,
    state: Manifest,
    model,
    tok,
    deltas: DeltaSource,
    cells: list[GenCell],
    summary: dict,
) -> None:
    """Batched generation over pending cells: cells sharing (layer, alpha,
    all_positions) run in ONE hooked generate_batch call per <=gen_batch chunk
    (per-row (B, H) deltas — never a batch-1 loop); unhooked cells batch
    together. Chunking is deterministic (sorted cell ids)."""
    pending = [c for c in cells if not state.done(c.cell_id)]
    summary["cells_skipped"] += len(cells) - len(pending)
    if not pending:
        return
    groups: dict[tuple, list[GenCell]] = {}
    for c in pending:
        key = ("hook", c.layer, c.alpha, c.all_positions) if c.delta_key else ("nohook",)
        groups.setdefault(key, []).append(c)
    for key in sorted(groups, key=repr):
        group = sorted(groups[key], key=lambda c: c.cell_id)
        for i in range(0, len(group), cfg.gen_batch):
            chunk = group[i : i + cfg.gen_batch]
            ctxs = [c.context for c in chunk]
            t0 = time.monotonic()
            if key[0] == "hook":
                _, layer, alpha, allpos = key
                dstack = torch.stack([deltas.resolve(c.delta_key, layer) for c in chunk])
                assert dstack.shape == (len(chunk), cfg.hidden), dstack.shape
                hook = DeltaHook(model, layer, dstack, alpha, all_positions=allpos)
                with hook:
                    outs = generate_batch(
                        model,
                        tok,
                        ctxs,
                        n=cfg.n_draws,
                        hook=hook,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                        seed_base=cfg.seed_base,
                    )
            else:
                outs = generate_batch(
                    model,
                    tok,
                    ctxs,
                    n=cfg.n_draws,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    seed_base=cfg.seed_base,
                )
            elapsed = time.monotonic() - t0
            sps = elapsed / (len(chunk) * cfg.n_draws)
            chunk_ids = [c.cell_id for c in chunk]
            logger.info(
                "[phase=%s] chunk B=%d n=%d %.1fs (%.2f s/sample) key=%s",
                chunk[0].phase,
                len(chunk),
                cfg.n_draws,
                elapsed,
                sps,
                key,
            )
            for cell, draws in zip(chunk, outs, strict=True):
                _persist_gen_cell(cfg, state, cell, draws, sps, chunk_ids)
                summary["cells_run"] += 1


# ── alpha selection (coherence gate) ──────────────────────────────────


def select_operating_alpha(
    flags_by_alpha: dict[float, list[bool]], grid: tuple[float, ...]
) -> float | None:
    """Largest grid alpha whose condition passes the >=50%-coherent gate
    (walking DOWN the geometric grid IS the 'lower alpha by x0.5 and retry'
    rule over already-run cells); None when every grid alpha fails."""
    for a in sorted(grid, reverse=True):
        if condition_passes(flags_by_alpha[a]):
            return a
    return None


def select_trait_alpha(
    pair_flags_by_alpha: dict[float, list[list[bool]]], grid: tuple[float, ...]
) -> float | None:
    """1d subset search: largest grid alpha whose coherence gate passes on
    >= half of the subset pairs; None when every grid alpha fails."""
    for a in sorted(grid, reverse=True):
        passes = [condition_passes(f) for f in pair_flags_by_alpha[a]]
        assert passes, "empty subset"
        if sum(passes) / len(passes) >= 0.5:
            return a
    return None


# ── phases ────────────────────────────────────────────────────────────


def phase_pilot(
    cfg: RunConfig, state: Manifest, model, tok, pairs: list[dict], summary: dict
) -> dict:
    """1 pair x alpha=1.0 x primary layer, replicated to B = cfg.gen_batch
    identical rows per hooked generate call — the SWEEP's chunk shape
    (``run_gen_cells`` chunks B cells per call; HF decode on A100-80 is
    memory-bandwidth-bound so per-step latency at B~8 ≈ B=1, and a batch-1
    pilot over-reads s/sample by ~B: the att-20260716-160022 gate false-fire)
    — x N draws (+ all-positions variant). Measures s/sample at the sweep
    shape and persists pilot.json (the full-sweep gate input). Row 0's draws
    stay the CANONICAL pilot draws (coherence_check + K2's pilot/std unit —
    semantics unchanged); all B rows' draws persist in the raw-completions
    pilot JSON (persist-by-default)."""
    pilot_path = cfg.out_root / "pilot.json"
    if state.done("pilot"):
        summary["cells_skipped"] += 1
        assert pilot_path.exists(), f"manifest marks pilot done but {pilot_path} missing"
        return json.loads(pilot_path.read_text())
    pair = pairs[0]
    cap = capture_vectors(model, tok, [pair["ctx_c"], pair["ctx_cprime"]], [cfg.primary_layer])
    rec_c, rec_cp = cap["per_context"]
    delta = rec_cp["v_c_context"][0] - rec_c["v_c_context"][0]
    assert delta.shape == (cfg.hidden,), delta.shape
    B = cfg.gen_batch
    contexts = [pair["ctx_c"]] * B
    # Per-row (B, H) delta stack — mirrors run_gen_cells' dstack contract so
    # DeltaHook exercises the identical batched shape it sees in the sweep.
    dstack = torch.stack([delta] * B)
    assert dstack.shape == (B, cfg.hidden), dstack.shape
    timings: dict[str, float] = {}
    coherence: dict[str, list[bool]] = {}
    for variant, allpos in (("std", False), ("allpos", True)):
        hook = DeltaHook(model, cfg.primary_layer, dstack, alpha=1.0, all_positions=allpos)
        t0 = time.monotonic()
        with hook:
            outs = generate_batch(
                model,
                tok,
                contexts,
                n=cfg.n_draws,
                hook=hook,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                seed_base=cfg.seed_base,
            )
        timings[variant] = time.monotonic() - t0
        assert len(outs) == B and all(len(rows) == cfg.n_draws for rows in outs), (
            len(outs),
            [len(rows) for rows in outs],
        )
        coherence[variant] = coherence_check(outs[0])  # row 0 = canonical pilot draws
        _write_json_atomic(
            cfg.bulk_root / "raw_completions" / "pilot" / f"{variant}.json",
            {
                "pair_id": pair["pair_id"],
                "variant": variant,
                "layer": cfg.primary_layer,
                "alpha": 1.0,
                "context": pair["ctx_c"],
                "pilot_batch": B,
                "canonical_row": 0,
                "draws": outs[0],
                "all_rows_draws": outs,
                "repro": _repro(cfg),
            },
        )
    n_samples = 2 * B * cfg.n_draws
    sps = sum(timings.values()) / n_samples
    pilot = {
        "pair_id": pair["pair_id"],
        "layer": cfg.primary_layer,
        "alpha": 1.0,
        "n_draws": cfg.n_draws,
        "max_new_tokens": cfg.max_new_tokens,
        "pilot_batch": B,
        "n_samples": n_samples,
        "timings_s": timings,
        "coherence_flags": coherence,
        "s_per_sample": sps,
        "threshold_s_per_sample": PILOT_MAX_S_PER_SAMPLE,
        "sweep_allowed": sps <= PILOT_MAX_S_PER_SAMPLE,
        "repro": _repro(cfg),
    }
    _write_json_atomic(pilot_path, pilot)
    logger.info(
        "[phase=pilot] s_per_sample=%.3f (threshold %.2f) sweep_allowed=%s B=%d",
        sps,
        PILOT_MAX_S_PER_SAMPLE,
        pilot["sweep_allowed"],
        B,
    )
    state.mark("pilot", {"s_per_sample": sps, "pilot_batch": B})
    summary["cells_run"] += 1
    return pilot


def _enforce_pilot_gate(cfg: RunConfig, pilot: dict) -> None:
    """DESIGNED HALT (never an anonymous crash) when the pilot exceeded the
    plan §9 per-sample budget: persist ``pilot_gate_report.json`` (the pilot
    dict + reason + the plan §13 descope-ladder pointer) into cfg.out_root and
    exit ``RC_PILOT_GATE`` so the dispatcher routes it on the artifact like
    K1/K2. ``--force`` overrides."""
    if pilot["s_per_sample"] <= PILOT_MAX_S_PER_SAMPLE or cfg.force:
        return
    reason = (
        f"pilot measured {pilot['s_per_sample']:.2f} s/sample at the sweep chunk shape "
        f"(B={pilot.get('pilot_batch', 1)}) > {PILOT_MAX_S_PER_SAMPLE} — refusing the full "
        "sweep (pass --force to override, or descope per the plan §13 ladder)"
    )
    report = {
        "criterion": "pilot timing gate: s_per_sample > plan §9 threshold",
        "fired": True,
        "reason": reason,
        "descope_pointer": "plan v5 §13 descope ladder",
        "pilot": pilot,
        "enforced": True,
    }
    _write_json_atomic(cfg.out_root / "pilot_gate_report.json", report)
    logger.error("[pilot_gate] %s", reason)
    raise SystemExit(RC_PILOT_GATE)


# ── kill criteria (plan v5 §3: K1 post-1a, K2 inside 1c) ──────────────


def _k1_null_p975(n_layers: int, hidden: int, n_draws: int, seed: int) -> float:
    """p97.5 of the max-over-layers cosine between two INDEPENDENT random unit
    directions in R^hidden — the quick CPU random-direction band the K1
    geometric half compares against (same max-over-layers selection shape as
    the per-pair statistic; selection-symmetric)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(n_draws, n_layers, hidden, generator=gen)
    v = torch.randn(n_draws, n_layers, hidden, generator=gen)
    cos = torch.nn.functional.cosine_similarity(u, v, dim=-1)  # (D, L)
    return float(torch.quantile(cos.max(dim=1).values, 0.975))


def _split_half_target_cos(rec_c: dict, rec_cp: dict) -> torch.Tensor | None:
    """Per-layer cosine between the two half-sample estimates of the answer
    target direction V_a(c') - V_a(c) (even vs odd kept draws). Returns None
    when either side has < 2 kept completions (stat not estimable)."""
    a_c = rec_c["v_a_per_completion"].float()  # (n_c, L, H)
    a_cp = rec_cp["v_a_per_completion"].float()  # (n_cp, L, H)
    if a_c.shape[0] < 2 or a_cp.shape[0] < 2:
        return None
    t1 = a_cp[0::2].mean(dim=0) - a_c[0::2].mean(dim=0)  # (L, H)
    t2 = a_cp[1::2].mean(dim=0) - a_c[1::2].mean(dim=0)
    return torch.nn.functional.cosine_similarity(t1, t2, dim=-1)  # (L,)


def evaluate_k1(cfg: RunConfig, pairs: list[dict]) -> dict:
    """K1 geometric half (plan v5 §3), evaluated post-1a on the 1a captures.

    A pair "shows separation" when its split-half answer-target cosine (max
    over the sweep layers) exceeds the random-direction band's p97.5 — a
    degenerate pair (V_a(c') ~ V_a(c) up to sampling noise) has two half
    estimates pointing in unrelated directions (cos within the band). K1
    fires when > K1_NO_SEP_FRAC of the evaluable pairs show NO separation.
    The judge half (ceiling judge-shift < 5 pts) is DEFERRED to the off-pod
    judge phase (issue1415_judge.k1_judge_check); the in-driver abort keys on
    the geometric half alone (recorded deviation, round-2 review)."""
    band_p975 = _k1_null_p975(len(cfg.layers), cfg.hidden, K1_NULL_DRAWS, K1_SEED)
    per_pair: dict[str, dict] = {}
    for p in pairs:
        pid = p["pair_id"]
        path = cfg.bulk_root / "activations" / f"{pid}.pt"
        assert path.exists(), f"K1 requires the phase-1a capture: {path}"
        blob = torch.load(path, map_location="cpu", weights_only=True)
        cos = _split_half_target_cos(blob["c"], blob["cprime"])
        if cos is None:
            per_pair[pid] = {"max_over_layers": None, "reason": "insufficient_kept_draws"}
            continue
        stat = float(cos.max())
        per_pair[pid] = {
            "split_half_cos_per_layer": [float(x) for x in cos],
            "max_over_layers": stat,
            "no_separation": stat <= band_p975,
        }
    evaluable = [v for v in per_pair.values() if v["max_over_layers"] is not None]
    n_no_sep = sum(v["no_separation"] for v in evaluable)
    frac = (n_no_sep / len(evaluable)) if evaluable else None
    fired = frac is not None and frac > K1_NO_SEP_FRAC
    return {
        "criterion": "K1 (geometric half): ceiling shows no answer-side separation",
        "statistic": (
            "split-half (even/odd draws) cosine of V_a(c')-V_a(c), max over sweep layers, "
            "vs p97.5 of the random-direction max-over-layers band"
        ),
        "judge_half": (
            "DEFERRED to the off-pod judge phase (issue1415_judge.k1_judge_check); "
            "this in-driver abort keys on the geometric half alone (recorded deviation)"
        ),
        "layers": list(cfg.layers),
        "null_band_p975": band_p975,
        "null_draws": K1_NULL_DRAWS,
        "null_seed": K1_SEED,
        "threshold_frac": K1_NO_SEP_FRAC,
        "n_pairs": len(pairs),
        "n_evaluable": len(evaluable),
        "n_no_separation": int(n_no_sep),
        "frac_no_separation": frac,
        "fired": bool(fired),
        "enforced": cfg.enforce_kill_criteria,
        "per_pair": per_pair,
    }


def evaluate_k2(cfg: RunConfig, pilot: dict, first_pairs: list[dict]) -> dict:
    """K2 (plan v5 §3): coherence collapse on the pilot + first-5-pairs units.

    Units = the pilot pair's std variant (single alpha=1.0 cell) + one unit
    per (extraction arm, first-5 pair) from the just-run primary-layer grid;
    a grid unit FAILS when NO grid alpha passes the >=50%-coherent gate (the
    x0.5 sub-grid retry is a later recovery step, deliberately outside the
    pre-registered "fails at ALL alpha values" clause). Fires when
    > K2_FAIL_FRAC of the units fail."""
    prim = cfg.primary_layer
    units: dict[str, dict] = {}
    pilot_pass = condition_passes(pilot["coherence_flags"]["std"])
    units["pilot/std"] = {"passes_any_alpha": pilot_pass}
    for arm in EXTRACTION_ARMS:
        for p in first_pairs:
            pid = p["pair_id"]
            passes = {
                _fmt(a): condition_passes(
                    load_cell_meta(cfg, f"gen1c/{arm}/{pid}/L{prim}/a{_fmt(a)}")["coherence_flags"]
                )
                for a in cfg.alpha_grid
            }
            units[f"{arm}/{pid}"] = {
                "pass_by_alpha": passes,
                "passes_any_alpha": any(passes.values()),
            }
    n_failed = sum(not u["passes_any_alpha"] for u in units.values())
    frac = n_failed / len(units)
    fired = frac > K2_FAIL_FRAC
    return {
        "criterion": "K2: coherence gate fails at ALL grid alphas on pilot + first-5 pairs",
        "primary_layer": prim,
        "alpha_grid": list(cfg.alpha_grid),
        "threshold_frac": K2_FAIL_FRAC,
        "n_units": len(units),
        "n_failed_all_alphas": int(n_failed),
        "frac_failed": frac,
        "fired": bool(fired),
        "enforced": cfg.enforce_kill_criteria,
        "units": units,
    }


def _enforce_kill(cfg: RunConfig, report: dict, name: str, rc: int) -> None:
    """Abort (distinct exit code) on a FIRED kill criterion; demoted to loud
    log lines under --tiny / --ignore-kill-criteria (verdict still persisted)."""
    if not report["fired"]:
        logger.info("[%s] verdict=PASS (%s)", name, report["criterion"])
        return
    logger.error("[%s] KILL CRITERION FIRED: %s", name, report["criterion"])
    logger.error(
        "[%s] report: %s",
        name,
        json.dumps({k: report[k] for k in report if k != "per_pair" and k != "units"}),
    )
    if cfg.enforce_kill_criteria:
        raise SystemExit(rc)
    logger.error("[%s] abort DEMOTED (tiny smoke / --ignore-kill-criteria) — continuing", name)


def phase_1b(
    cfg: RunConfig,
    state: Manifest,
    model,
    tok,
    deltas: DeltaSource,
    pairs: list[dict],
    summary: dict,
) -> None:
    """Unhooked generation under c (hf_nohook_base) and c' (ctx_swap_ceil)."""
    cells = []
    for p in pairs:
        for arm, ctx_key, label in (
            ("c", "ctx_c", "hf_nohook_base"),
            ("cprime", "ctx_cprime", "ctx_swap_ceil"),
        ):
            cells.append(
                GenCell(
                    cell_id=f"gen1b/{p['pair_id']}/{arm}",
                    phase="phase1b",
                    pair_id=p["pair_id"],
                    context=p[ctx_key],
                    extra={"arm_label": label},
                )
            )
    run_gen_cells(cfg, state, model, tok, deltas, cells, summary)


def phase_1a(cfg: RunConfig, state: Manifest, model, tok, pairs: list[dict], summary: dict) -> None:
    """Capture V_c (prefix + context arms) + V_a over the 1b completions."""
    for p in pairs:
        pid = p["pair_id"]
        cell_id = f"capture1a/{pid}"
        if state.done(cell_id):
            summary["cells_skipped"] += 1
            continue
        draws_c = _load_draws(cfg, f"gen1b/{pid}/c")
        draws_cp = _load_draws(cfg, f"gen1b/{pid}/cprime")
        out = capture_vectors(
            model,
            tok,
            [p["ctx_c"], p["ctx_cprime"]],
            list(cfg.layers),
            completions=[draws_c, draws_cp],
            batch_size=cfg.capture_batch,
        )
        rec_c, rec_cp = out["per_context"]
        for rec in (rec_c, rec_cp):
            assert rec["v_c_context"].shape == (len(cfg.layers), cfg.hidden)
        _save_pt_atomic(
            cfg.bulk_root / "activations" / f"{pid}.pt",
            {
                "pair_id": pid,
                "layers": list(cfg.layers),
                "c": rec_c,
                "cprime": rec_cp,
                "repro": _repro(cfg),
            },
        )
        _write_json_atomic(
            _cell_meta_path(cfg, cell_id),
            {
                "cell_id": cell_id,
                "phase": "phase1a",
                "pair_id": pid,
                "layers": list(cfg.layers),
                "n_empty_completions": {
                    "c": rec_c["n_empty_completions"],
                    "cprime": rec_cp["n_empty_completions"],
                },
                "tensor_file": f"activations/{pid}.pt",
                "repro": _repro(cfg),
            },
        )
        state.mark(cell_id, {"phase": "phase1a"})
        summary["cells_run"] += 1
        logger.info("[phase=capture1a] %s captured", pid)


def _grid_cells(cfg: RunConfig, pairs_subset: list[dict]) -> list[GenCell]:
    """Primary-layer full-alpha-grid cells for a pair subset (both arms)."""
    prim = cfg.primary_layer
    return [
        GenCell(
            cell_id=f"gen1c/{arm}/{p['pair_id']}/L{prim}/a{_fmt(a)}",
            phase="phase1c_grid",
            pair_id=p["pair_id"],
            context=p["ctx_c"],
            layer=prim,
            alpha=a,
            delta_key=("pair", p["pair_id"], arm),
            extra={"extraction_arm": arm},
        )
        for arm in EXTRACTION_ARMS
        for p in pairs_subset
        for a in cfg.alpha_grid
    ]


def phase_1c(
    cfg: RunConfig,
    state: Manifest,
    model,
    tok,
    deltas: DeltaSource,
    pairs: list[dict],
    summary: dict,
    pilot: dict,
) -> dict:
    """Delta-addition sweep: primary-layer alpha grid — pilot + FIRST 5 pairs
    first, then the K2 coherence-collapse gate (halt on fire), then the
    remaining pairs — coherence alpha selection (with one x0.5 sub-grid
    retry), remaining layers at the operating alpha, all-positions variant at
    the primary layer."""
    prim = cfg.primary_layer
    first = pairs[: min(K2_FIRST_PAIRS, len(pairs))]
    rest = pairs[len(first) :]

    # K2 ordering (plan §3, round-2 fix): gate on pilot + first-5 coherence
    # BEFORE the remaining ~23-pair x 4-alpha grid burns GPU time.
    run_gen_cells(cfg, state, model, tok, deltas, _grid_cells(cfg, first), summary)
    k2 = evaluate_k2(cfg, pilot, first)
    _write_json_atomic(cfg.out_root / "k2_report.json", k2)
    summary["k2"] = {"fired": k2["fired"], "frac_failed": k2["frac_failed"]}
    _enforce_kill(cfg, k2, "k2", RC_K2_HALT)

    run_gen_cells(cfg, state, model, tok, deltas, _grid_cells(cfg, rest), summary)

    retry_alpha = min(cfg.alpha_grid) / 2.0  # the gate's one sub-grid x0.5 retry
    selection: dict[str, dict] = {}
    retry_cells: list[GenCell] = []
    for arm in EXTRACTION_ARMS:
        for p in pairs:
            pid = p["pair_id"]
            flags = {
                a: load_cell_meta(cfg, f"gen1c/{arm}/{pid}/L{prim}/a{_fmt(a)}")["coherence_flags"]
                for a in cfg.alpha_grid
            }
            op = select_operating_alpha(flags, cfg.alpha_grid)
            selection[f"{arm}/{pid}"] = {
                "operating_alpha": op,
                "pass_by_alpha": {_fmt(a): condition_passes(f) for a, f in flags.items()},
                "retried": False,
            }
            if op is None:
                retry_cells.append(
                    GenCell(
                        cell_id=f"gen1c_retry/{arm}/{pid}/L{prim}/a{_fmt(retry_alpha)}",
                        phase="phase1c_retry",
                        pair_id=pid,
                        context=p["ctx_c"],
                        layer=prim,
                        alpha=retry_alpha,
                        delta_key=("pair", pid, arm),
                        extra={"extraction_arm": arm},
                    )
                )
    if retry_cells:
        run_gen_cells(cfg, state, model, tok, deltas, retry_cells, summary)
        for c in retry_cells:
            key = f"{c.extra['extraction_arm']}/{c.pair_id}"
            passed = condition_passes(load_cell_meta(cfg, c.cell_id)["coherence_flags"])
            selection[key]["retried"] = True
            selection[key]["operating_alpha"] = retry_alpha if passed else None
    _write_json_atomic(
        cfg.out_root / "alpha_selection_1c.json",
        {
            "primary_layer": prim,
            "grid": list(cfg.alpha_grid),
            "retry_alpha": retry_alpha,
            "selection": selection,
            "repro": _repro(cfg),
        },
    )

    rem_cells: list[GenCell] = []
    ap_cells: list[GenCell] = []
    for arm in EXTRACTION_ARMS:
        for p in pairs:
            pid = p["pair_id"]
            op = selection[f"{arm}/{pid}"]["operating_alpha"]
            if op is None:  # never coherent, even at the retry alpha — recorded, skipped
                continue
            for layer in cfg.layers:
                if layer == prim:
                    continue
                rem_cells.append(
                    GenCell(
                        cell_id=f"gen1c/{arm}/{pid}/L{layer}/a{_fmt(op)}",
                        phase="phase1c_layers",
                        pair_id=pid,
                        context=p["ctx_c"],
                        layer=layer,
                        alpha=op,
                        delta_key=("pair", pid, arm),
                        extra={"extraction_arm": arm},
                    )
                )
            ap_cells.append(
                GenCell(
                    cell_id=f"gen1c_allpos/{arm}/{pid}/L{prim}/a{_fmt(op)}",
                    phase="phase1c_allpos",
                    pair_id=pid,
                    context=p["ctx_c"],
                    layer=prim,
                    alpha=op,
                    all_positions=True,
                    delta_key=("pair", pid, arm),
                    extra={"extraction_arm": arm},
                )
            )
    run_gen_cells(cfg, state, model, tok, deltas, rem_cells, summary)
    run_gen_cells(cfg, state, model, tok, deltas, ap_cells, summary)
    return selection


def phase_1d(
    cfg: RunConfig,
    state: Manifest,
    model,
    tok,
    deltas: DeltaSource,
    pairs: list[dict],
    summary: dict,
) -> dict:
    """r_B arm: alpha search on a 5-pair subset, then the full pair set x
    traits at the per-trait selected alpha."""
    prim = cfg.primary_layer
    subset = pairs[: min(5, len(pairs))]
    search_cells = [
        GenCell(
            cell_id=f"gen1d_search/{trait}/{p['pair_id']}/a{_fmt(a)}",
            phase="phase1d_search",
            pair_id=p["pair_id"],
            context=p["ctx_c"],
            layer=prim,
            alpha=a,
            delta_key=("rb", trait),
            extra={"trait": trait},
        )
        for trait in TRAITS
        for p in subset
        for a in cfg.alpha_grid
    ]
    run_gen_cells(cfg, state, model, tok, deltas, search_cells, summary)

    retry_alpha = min(cfg.alpha_grid) / 2.0
    selection: dict[str, dict] = {}
    for trait in TRAITS:
        by_alpha = {
            a: [
                load_cell_meta(cfg, f"gen1d_search/{trait}/{p['pair_id']}/a{_fmt(a)}")[
                    "coherence_flags"
                ]
                for p in subset
            ]
            for a in cfg.alpha_grid
        }
        sel_a = select_trait_alpha(by_alpha, cfg.alpha_grid)
        selection[trait] = {
            "selected_alpha": sel_a,
            "retried": False,
            "subset_pair_ids": [p["pair_id"] for p in subset],
        }
        if sel_a is None:
            rcells = [
                GenCell(
                    cell_id=f"gen1d_retry/{trait}/{p['pair_id']}/a{_fmt(retry_alpha)}",
                    phase="phase1d_retry",
                    pair_id=p["pair_id"],
                    context=p["ctx_c"],
                    layer=prim,
                    alpha=retry_alpha,
                    delta_key=("rb", trait),
                    extra={"trait": trait},
                )
                for p in subset
            ]
            run_gen_cells(cfg, state, model, tok, deltas, rcells, summary)
            passes = [
                condition_passes(load_cell_meta(cfg, c.cell_id)["coherence_flags"]) for c in rcells
            ]
            selection[trait]["retried"] = True
            if sum(passes) / len(passes) >= 0.5:
                selection[trait]["selected_alpha"] = retry_alpha
    _write_json_atomic(
        cfg.out_root / "alpha_selection_1d.json",
        {
            "primary_layer": prim,
            "grid": list(cfg.alpha_grid),
            "retry_alpha": retry_alpha,
            "selection": selection,
            "repro": _repro(cfg),
        },
    )

    full_cells = [
        GenCell(
            cell_id=f"gen1d_full/{trait}/{p['pair_id']}/a{_fmt(selection[trait]['selected_alpha'])}",
            phase="phase1d_full",
            pair_id=p["pair_id"],
            context=p["ctx_c"],
            layer=prim,
            alpha=selection[trait]["selected_alpha"],
            delta_key=("rb", trait),
            extra={"trait": trait},
        )
        for trait in TRAITS
        if selection[trait]["selected_alpha"] is not None
        for p in pairs
    ]
    run_gen_cells(cfg, state, model, tok, deltas, full_cells, summary)
    return selection


# ── phase 1e: steered-completions V_a capture (plan §4.10 DV (a) input) ─


def _steered_cell_metas(cfg: RunConfig) -> list[dict]:
    """Every persisted steered-cell metadata record (nested under out_root/cells)."""
    cells_dir = cfg.out_root / "cells"
    assert cells_dir.exists(), f"phase-1 cells metadata missing: {cells_dir}"
    metas = [json.loads(p.read_text()) for p in sorted(cells_dir.rglob("*.json"))]
    return [m for m in metas if m.get("phase") in CAPTURED_PHASES]


def phase_1e(cfg: RunConfig, state: Manifest, model, tok, pairs: list[dict], summary: dict) -> None:
    """Teacher-forced V_a capture over EVERY persisted steered completion cell.

    Per cell: ``capture_vectors`` over the cell's draws (batched across draws,
    all sweep layers in one forward each) -> ``activations_steered/<cell_id>.pt``
    with ``v_a_mean`` (L, H), ``v_a_per_completion`` and the empty-completion
    count (the 1a drop-with-record contract carried forward). A cell whose
    draws are ALL empty is recorded (``all_empty: true``) and skipped — the
    projection driver excludes it with the recorded reason. Then writes the
    map-transport canonical ``<pair_id>__<arm>.pt`` files (selected
    operating-alpha cell at the primary layer)."""
    ctx_by_pair = {p["pair_id"]: p["ctx_c"] for p in pairs}
    metas = _steered_cell_metas(cfg)
    assert metas, "phase-1e: no steered cells found (1c/1d must run first)"
    steered_root = cfg.bulk_root / "activations_steered"
    for meta in metas:
        cell_id = meta["cell_id"]
        cap_id = f"capture1e/{cell_id}"
        if state.done(cap_id):
            summary["cells_skipped"] += 1
            continue
        draws = _load_draws(cfg, cell_id)
        common_fields = {
            "cell_id": cell_id,
            "pair_id": meta["pair_id"],
            "phase": meta["phase"],
            "layer": meta["layer"],
            "alpha": meta["alpha"],
            "all_positions": meta["all_positions"],
            "extraction_arm": meta.get("extraction_arm"),
            "trait": meta.get("trait"),
            "layers": list(cfg.layers),
            "repro": _repro(cfg),
        }
        out_path = steered_root / f"{cell_id}.pt"
        # Mirror capture_vectors' empty-completion criterion (zero tokens) so
        # an all-empty cell is a recorded skip, never a crash.
        n_tokens = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in draws]
        if not any(n_tokens):
            _save_pt_atomic(out_path, {**common_fields, "all_empty": True})
            state.mark(cap_id, {"phase": "phase1e", "all_empty": True})
            summary["cells_run"] += 1
            logger.warning(
                "[phase=capture1e] %s: ALL %d draws empty — capture skipped (recorded)",
                cell_id,
                len(draws),
            )
            continue
        cap = capture_vectors(
            model,
            tok,
            [ctx_by_pair[meta["pair_id"]]],
            list(cfg.layers),
            completions=[draws],
            batch_size=cfg.capture_batch,
        )
        rec = cap["per_context"][0]
        assert rec["v_a_mean"].shape == (len(cfg.layers), cfg.hidden), rec["v_a_mean"].shape
        _save_pt_atomic(
            out_path,
            {
                **common_fields,
                "all_empty": False,
                "v_a_mean": rec["v_a_mean"],
                "v_a_per_completion": rec["v_a_per_completion"],
                "n_empty_completions": rec["n_empty_completions"],
            },
        )
        state.mark(cap_id, {"phase": "phase1e", "n_empty": rec["n_empty_completions"]})
        summary["cells_run"] += 1
        logger.info(
            "[phase=capture1e] %s captured (n_empty=%d)", cell_id, rec["n_empty_completions"]
        )
    _write_canonical_steered(cfg, state, pairs, summary, steered_root)


def _write_canonical_steered(
    cfg: RunConfig, state: Manifest, pairs: list[dict], summary: dict, steered_root: Path
) -> None:
    """Map-transport contract: ``activations_steered/<pair_id>__<arm>.pt`` =
    the SELECTED cell's capture (operating alpha at the primary layer; the
    retry cell when the selection retried). Coherence-failed pairs (operating
    alpha None) are skipped WITH a recorded reason (plan §8: excluded from the
    geometric DV but reported)."""
    sel_blob = json.loads((cfg.out_root / "alpha_selection_1c.json").read_text())
    selection, retry_alpha = sel_blob["selection"], sel_blob["retry_alpha"]
    index: dict[str, dict] = {}
    for arm in EXTRACTION_ARMS:
        for p in pairs:
            pid = p["pair_id"]
            key = f"{arm}/{pid}"
            op = selection[key]["operating_alpha"]
            if op is None:
                index[key] = {"skipped": "coherence_failed_all_alpha"}
                continue
            prefix = "gen1c_retry" if (selection[key]["retried"] and op == retry_alpha) else "gen1c"
            src_cell = f"{prefix}/{arm}/{pid}/L{cfg.primary_layer}/a{_fmt(op)}"
            index[key] = {"canonical_of": src_cell, "alpha": op}
            mark_id = f"capture1e_canonical/{arm}/{pid}"
            if state.done(mark_id):
                summary["cells_skipped"] += 1
                continue
            src = steered_root / f"{src_cell}.pt"
            assert src.exists(), f"canonical steered capture source missing: {src}"
            blob = torch.load(src, map_location="cpu", weights_only=True)
            assert not blob.get("all_empty"), (
                f"selected cell {src_cell} is all-empty — cannot be canonical "
                "(it passed the coherence gate, so this indicates a capture bug)"
            )
            _save_pt_atomic(steered_root / f"{pid}__{arm}.pt", {**blob, "canonical_of": src_cell})
            state.mark(mark_id, {"phase": "phase1e", "canonical_of": src_cell})
            summary["cells_run"] += 1
    _write_json_atomic(
        cfg.out_root / "steered_canonical_index.json",
        {"index": index, "primary_layer": cfg.primary_layer, "repro": _repro(cfg)},
    )


# ── top-level driver ──────────────────────────────────────────────────


def run_phase1(cfg: RunConfig) -> dict:
    """Run (or resume) the phase-1 pipeline; returns the run summary."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.bulk_root.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(cfg)
    state = Manifest.load_or_init(
        cfg.out_root / "phase1_manifest.json", _regime(cfg, _sha256(cfg.pair_bank_path))
    )
    if cfg.upload_mode == "hf":
        assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing — required for --upload hf"
    summary: dict = {"cells_run": 0, "cells_skipped": 0, "uploads": 0, "uploads_skipped": 0}
    deltas = DeltaSource(cfg)
    if not cfg.pilot_only:
        deltas.preload_rb()  # fail EARLY on the 1d inputs, before model load / GPU spend
    model, tok = load_model_and_tokenizer(cfg)

    pilot = phase_pilot(cfg, state, model, tok, pairs, summary)
    _upload_phase(cfg, state, summary, "raw_completions/pilot", f"{RAW_PREFIX}/pilot")
    summary["pilot"] = {
        "s_per_sample": pilot["s_per_sample"],
        "sweep_allowed": pilot["sweep_allowed"],
    }
    if cfg.pilot_only:
        return summary
    _enforce_pilot_gate(cfg, pilot)

    phase_1b(cfg, state, model, tok, deltas, pairs, summary)
    _upload_phase(cfg, state, summary, "raw_completions/gen1b", f"{RAW_PREFIX}/gen1b")

    phase_1a(cfg, state, model, tok, pairs, summary)
    _upload_phase(cfg, state, summary, "activations", TENSOR_PREFIX)

    # K1 (plan §3): the ceiling must show answer-side separation, else the
    # pair bank carries no usable signal — abort BEFORE the ~10 GPU-h sweep.
    k1 = evaluate_k1(cfg, pairs)
    _write_json_atomic(cfg.out_root / "k1_report.json", k1)
    summary["k1"] = {"fired": k1["fired"], "frac_no_separation": k1["frac_no_separation"]}
    _enforce_kill(cfg, k1, "k1", RC_K1_ABORT)

    summary["alpha_selection_1c"] = phase_1c(cfg, state, model, tok, deltas, pairs, summary, pilot)
    for sub in ("gen1c", "gen1c_retry", "gen1c_allpos"):
        _upload_phase(cfg, state, summary, f"raw_completions/{sub}", f"{RAW_PREFIX}/{sub}")

    summary["alpha_selection_1d"] = phase_1d(cfg, state, model, tok, deltas, pairs, summary)
    for sub in ("gen1d_search", "gen1d_retry", "gen1d_full"):
        _upload_phase(cfg, state, summary, f"raw_completions/{sub}", f"{RAW_PREFIX}/{sub}")

    phase_1e(cfg, state, model, tok, pairs, summary)
    _upload_phase(cfg, state, summary, "activations_steered", STEERED_TENSOR_PREFIX)

    logger.info(
        "[phase=done] cells_run=%d cells_skipped=%d uploads=%d",
        summary["cells_run"],
        summary["cells_skipped"],
        summary["uploads"],
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    cfg = build_config(parse_args(argv))
    summary = run_phase1(cfg)
    print(
        json.dumps(
            {
                "cells_run": summary["cells_run"],
                "cells_skipped": summary["cells_skipped"],
                "uploads": summary["uploads"],
                "pilot_s_per_sample": summary["pilot"]["s_per_sample"],
                "sweep_allowed": summary["pilot"]["sweep_allowed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
