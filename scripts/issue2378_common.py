"""Shared constants + helpers for issue #2378 (cross-framing context->answer transfer).

Consumed by ``scripts/issue2378_gen.py`` and ``scripts/issue2378_judge.py``
(and by the later capture/fits drivers). Everything here is deliberately
light-import: no torch/vllm/transformers at module top, so VM phases
(pool/bank building, judge waves) run on the repo venv while pod phases defer
their heavy imports into function bodies.

Key contracts (plan v7, tasks/running/2378/plans/plan.md):
- Cells (ACTIVE panel, plan v7 Amendment record A / epm:progress v70): chat,
  plain_text, 5 story-Q cells, 2 user arms (11 cells). The 4 dialogue cells
  are DESCOPED from every active enumeration; their banks/constants stay
  defined (inert) for tests + archival r1/r2 artifact readers.
- Sampling pins: temperature 1.0, top_p 0.95, top_k 20, seed 137
  (fresh draws 138-141); ``enable_thinking=False`` everywhere a chat template
  is rendered.
- Caps: SegA 512 (cap-hit exempt from regen), SegB 1024, chat/plain 2048,
  sim-user 1024; cap-hit > 2%/cell => regen the cap-hit rows at 2x cap.
- Content hygiene: real-corpus text (LMSYS/WildChat rows) is NEVER printed --
  digest-only logging via :func:`text_digest`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Iterator
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ISSUE = 2378
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue2378_xframing"
MODEL_ID = "Qwen/Qwen3.6-27B"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Model venv (pods) — plan §10 Repro card "Env (model venv, pods)". The repo
# venv (uv.lock: vLLM 0.11.0 / transformers 4.57.6) cannot load model type
# `qwen3_5` (the P1 engine-init crash, r5 fix), so every MODEL step runs under
# a DEDICATED pod venv. Exact pins resolved on pod-2378 at P1 (host driver
# 580.159.04 -> CUDA-13-native wheels): vllm==0.27.1 itself pins
# torch==2.13.0; transformers==5.15.1 is the qwen3_5-bearing release.
# python-dotenv (uv.lock pin) is the one extra pure-python dist the model-venv
# import path needs (orchestrate/env.py module-top `from dotenv import ...`).
# Repo env unchanged for the non-model phases (plan: P0/P6/P7 + judge/uploads).
MODEL_VENV_DEFAULT = "/root/eps-model-venv"
MODEL_VENV_PINS = {"vllm": "0.27.1", "transformers": "5.15.1", "torch": "2.13.0"}
MODEL_VENV_EXTRA_PINS = ("python-dotenv==1.2.2",)
# Dists that must be ABSENT from the model venv (r7 crash-fix, epm:failure v3,
# assert_tag flashinfer-py311-array-subscript): vllm 0.27.1 HARD-pins
# flashinfer-python==0.6.16.post3 in its own requires_dist (PyPI, verified
# 2026-08-20), whose comm/fd_exchange.py:55 evaluates the annotation
# `array.array[int]` at import time — `array.array` is subscriptable only on
# py>=3.13, so the venv's py3.11 raises TypeError. vLLM imports flashinfer
# lazily inside the compile backend (passes/fusion/allreduce_rms_fusion.py:90)
# behind an ImportError-ONLY guard, so the TypeError ESCAPES and kills
# EngineCore ~30 s into engine init (all 4 p1.sega shards). REMOVAL is the
# minimal reliable fix: every vLLM flashinfer call site is find_spec-guarded
# when the dist is ABSENT, and TP=1 shards never need allreduce fusion. A
# "pin a compatible version" fix is NOT available — vllm's own exact
# `==0.6.16.post3` pin makes any other flashinfer-python version a resolver
# conflict. Because `uv pip install vllm==0.27.1` re-resolves the closure and
# re-adds the dist, _build_model_venv's uninstall step runs AFTER install on
# EVERY build/repair (uv pip uninstall is a clean rc=0 no-op when absent).
# Maps dist name -> top-level import name (the find_spec probe target).
MODEL_VENV_BANNED_DISTS = {"flashinfer-python": "flashinfer"}
# Env pins injected into EVERY dispatcher-launched step env (r8 crash-fix,
# epm:failure v4, assert_tag flashinfer-absent-sampler-probe-modulenotfound):
# with flashinfer-python REMOVED from the model venv (r7, above), vllm 0.27.1
# still probes the flashinfer SAMPLER by DEFAULT at EngineCore init —
# envs.py:848-852 (tag v0.27.1, verified 2026-08-21): unset ->
# VLLM_USE_FLASHINFER_SAMPLER=True; the probe docstring "Assumes flashinfer is
# installed, as guaranteed by requirements/cuda.txt" — and dies on the bare
# `from flashinfer import ...` at vllm/v1/attention/backends/flashinfer.py:12
# (reached via TopKTopPSampler.__init__ -> flashinfer_sampler_supported(),
# topk_topp_sampler.py:96-98; killed all 4 p1.sega shards ~40 s in). "0"
# short-circuits BEFORE that import: topk_topp_sampler.py:45-51 checks
# `if not envs.VLLM_USE_FLASHINFER_SAMPLER: return False` ahead of the
# line-51 backend import, and envs.py parses "0" -> bool(int("0")) -> False.
# Injected AUTHORITATIVELY (after os.environ) in Runner.run/fanout/parallel:
# inert for non-vllm steps; an inherited =1 would deterministically crash
# engine init on the flashinfer-free venv, so the pin always wins.
LAUNCH_ENV_PINS = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
# Engine-kwarg pins injected into EVERY create_vllm_engine call (the shared
# seam both the engine_smoke gate and the p1/p2/p4 gen shards construct
# through). r9 crash-fix (epm:failure v5, assert_tag
# flashinfer-absent-gdn-prefill-modulenotfound): Qwen3.6-27B is hybrid-GDN
# (text_config layer_types = linear_attention + full_attention every 4th);
# vllm 0.27.1's GDN prefill resolver
# (model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:85-133, tag
# v0.27.1 = 6e448d0ea9bf, verified 2026-08-21) reads
# additional_config["gdn_prefill_backend"] (default "auto") and on SM90
# (H200) auto-selects "flashinfer" UNCONDITIONALLY — no availability check,
# the ONLY unguarded flashinfer auto-select reachable for this model — then
# hard-imports flashinfer.gdn_prefill inside fi_chunk_gated_delta_rule
# (:174) at the FIRST prefill: a forward-time ModuleNotFoundError on the
# flashinfer-free venv (r7 ban), unreachable by the r8 sampler ENV pin
# (different subsystem). NO env-var route exists for this knob — it threads
# ONLY as the EngineArgs dataclass field `gdn_prefill_backend`
# (engine/arg_utils.py:752 -> additional_config, :2459-2460), i.e. an
# LLM(...) engine kwarg via create_vllm_engine's **kwargs. "triton" routes
# to the IN-TREE vllm.third_party.flash_linear_attention kernels
# (qwen_gdn_linear_attn.py:282 — no new dependency; vllm's own in-log hint:
# "Set --gdn-prefill-backend triton to skip JIT"). The same resolver also
# feeds the GDN attention metadata builder
# (v1/attention/backends/gdn_attn.py:99-104), so one pin covers both
# consumers. Passed UNGUARDED by design (no EngineArgs-field introspection,
# unlike the language_model_only OPTIMIZATION): an engine whose EngineArgs
# lacks the field raises a loud TypeError at construction — never a silent
# skip of a load-bearing pin.
ENGINE_KWARG_PINS = {"gdn_prefill_backend": "triton"}
MODEL_PY_ENV = "EPM_I2378_MODEL_PY"  # explicit model-interpreter override
# Host-driver floor for the CUDA-13 wheel stack above (torch 2.13.0 ships
# cu130-linked binaries; vllm 0.27.1 links libcudart.so.13). A pre-580 host
# driver passes every CPU-only gate and dies at vLLM engine init — the #2330
# crash shape (gotchas.md "RunPod host driver vs CUDA-major wheel mismatch").
# The forward-compat escape is cuda-compat-13-0 + LD_LIBRARY_PATH.
MODEL_DRIVER_FLOOR_MAJOR = 580
CUDA_COMPAT_DIR = "/usr/local/cuda-13.0/compat"
SKIP_DRIVER_PROBE_ENV = "EPM_I2378_SKIP_DRIVER_PROBE"  # deliberate waiver, logged loud

SEED = 137
FRESH_SEEDS = (138, 139, 140, 141)
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20

SEGA_MAX_TOKENS = 512
SEGB_MAX_TOKENS = 1024
CHAT_MAX_TOKENS = 2048
PLAIN_MAX_TOKENS = 2048
USER_SIM_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS = 1024

CAP_HIT_REGEN_THRESHOLD = 0.02  # > 2%/cell => regen cap-hit rows at 2x cap (SegA exempt)
# G1 recalibration (r11): 250 -> 512, the full SegA generation cap. The 250-tok
# window was the plan's LOW-confidence carry-over premise (plan v6 §12.7,
# plan:447); the round-1 pilot measured 236q/288d no_quote_in_window rejects
# against 512-token generations, so the window now covers the whole generation
# (gen.py's offset slicing degrades to len(text) when offs <= window).
MINER_WINDOW_TOKENS = 512

PLAIN_STOP = ["\nUser:", "\n\nUser:"]
CHAT_STOP = ["<|im_end|>"]
USER_SIM_STOP = ["<|im_end|>", "<|im_start|>"]
USER_TURN_HEADER = "<|im_start|>user\n"

# ---------------------------------------------------------------------------
# Character panel: #1310/#1345 personas verbatim (scripts/issue1310_common.py
# lines 128-133; build_banks asserts parity against the live module) + Astra
# (plan §4.1, new assistant-like character).
# ---------------------------------------------------------------------------

PERSONAS: dict[str, str] = {
    "Wren": "a warm, endlessly helpful assistant who patiently helps anyone who asks",
    "HELIOS": "a calm, precise artificial intelligence",
    "Dana": "an ordinary, unremarkable everyday person",
    "Vex": "a theatrical, scheming villain who delights in menace",
    "Astra": "an advanced AI assistant, precise and even-tempered, who knows it is an AI",
}

STORY_Q_CELLS: tuple[str, ...] = (
    "storyq_astra",
    "storyq_helios",
    "storyq_wren",
    "storyq_dana",
    "storyq_vex",
)
# DESCOPED (plan v7 Amendment record A, epm:progress v70 clause 1): the
# dialogue family is dropped from the ACTIVE panel — DIALOG_CELLS stays
# DEFINED (tests + archival r1/r2 artifact readers) but appears in NO active
# enumeration (STORY_CELLS / ALL_CELLS / ACTIVE_FAMILIES below exclude it).
DIALOG_CELLS: tuple[str, ...] = (
    "dialog_astra",
    "dialog_helios",
    "dialog_dana",
    "dialog_vex",
)  # no Wren dialogue cell (plan §5)
# ACTIVE story panel (v7): question-family only. Pre-v7 this was
# STORY_Q_CELLS + DIALOG_CELLS; every consumer that enumerates "the story
# cells to generate / judge / capture / fit" reads THIS switch.
STORY_CELLS: tuple[str, ...] = STORY_Q_CELLS
USER_CELLS: tuple[str, ...] = ("chat_user_real", "chat_user_sim")
ALL_CELLS: tuple[str, ...] = ("chat", "plain_text") + STORY_CELLS + USER_CELLS  # 11 active
# ACTIVE mining/judging families (v7): question only. The G1 gate, the
# admission-slice balancer, the judge pilot sampler, and the P2 wave sizing
# all iterate THIS tuple — dialogue never enters an active family loop.
ACTIVE_FAMILIES: tuple[str, ...] = ("question",)
FAMILY_CELLS: dict[str, tuple[str, ...]] = {"question": STORY_Q_CELLS}

_CHAR_BY_SLUG = {
    "astra": "Astra",
    "helios": "HELIOS",
    "wren": "Wren",
    "dana": "Dana",
    "vex": "Vex",
}
# Interpretation maps deliberately cover the FULL question+dialogue panel:
# archival readers (the P1R digest re-compose pools the r2 dialogue rows
# for-the-record; gen's salvage/probe fixtures) still resolve dialog_* cells.
CELL_CHARACTER: dict[str, str] = {
    c: _CHAR_BY_SLUG[c.split("_", 1)[1]] for c in STORY_Q_CELLS + DIALOG_CELLS
}
CELL_FAMILY: dict[str, str] = {
    **{c: "question" for c in STORY_Q_CELLS},
    **{c: "dialogue" for c in DIALOG_CELLS},
}

# ---------------------------------------------------------------------------
# Data pools (plan §4.1): pinned #1738 sampling manifest + draw targets +
# question / user-turn filters (#2054 measured constants + English-script).
# ---------------------------------------------------------------------------

MANIFEST_PREFIX = "issue1738_multiturn/sampling_manifest"
MANIFEST_REVISION = "003e392548fcbbe866c6f345f4688d8176cd9f04"

CHAT_DRAW_N = 12_000
PLAIN_DRAW_N = 10_000
USER_DRAW_N = 10_000

QUESTION_MIN_CHARS = 16  # Source: #2054 issue2054_phase_a.py measured constants
QUESTION_MAX_CHARS = 400
USER_TURN_MIN_CHARS = 16  # plan §4.2b measured-turn band, both user arms
USER_TURN_MAX_CHARS = 2_000

STORY_TARGET_KEPT = 8_000
CHAT_TARGET_KEPT = 9_000
FLOOR_KEPT = 6_500  # binding per-cell floor (81.25%)
CLOSE_MISS_FLOOR = 5_850  # >= 90% of floor => one recorded escalation wave

# Default local roots (pod dispatch overrides via CLI flags).
POOLS_DIR = REPO_ROOT / "data" / "issue_2378" / "pools"
BANKS_DIR = REPO_ROOT / "data" / "issue_2378" / "banks"
LEDGER_ROOT = REPO_ROOT / "eval_results" / "issue_2378"
RAW_ROOT_DEFAULT = REPO_ROOT / "data" / "issue_2378" / "raw_completions"
# Pilot capture store default (capture.py --pilot-out-root). Plan §10 declared
# discard: all-layer chat states, regenerable from the persisted chat text.
PILOT_STORE_DEFAULT = REPO_ROOT / "data" / "issue_2378" / "activations_pilot"


def pilot_capture_out_root(rnd: int, stable: Path | None = None) -> Path:
    """Round-scoped pilot capture out-root (plan §4.7 out-root fix, r12).

    Round 1 keeps the stable default (byte-identical to the pre-fix path);
    round >= 2 gets the SIBLING dir ``activations_pilot_r{rnd}`` so a fresh
    round never lands in a prior round's StageLedger (the r2 crash: capture
    into round 1's store tripped the ledger regime-fingerprint fail-loud).
    A sibling (not a subdir) keeps the store root self-contained for the
    ledger's part-file scan.
    """
    base = Path(stable) if stable is not None else PILOT_STORE_DEFAULT
    if rnd <= 1:
        return base
    return base.parent / f"{base.name}_r{rnd}"


# Raw-completion stage names (plan §10 prefixes + judge_congruence,
# persist-by-default addition for the congruence wave's raw judge rows;
# judge_*_pilot added r2 — pilot rows persist like production waves).
RAW_STAGES: tuple[str, ...] = (
    "sega",
    "sega_mined",
    "judge_admission",
    "judge_congruence",
    "judge_admission_pilot",
    "judge_congruence_pilot",
    "segb",
    "chat",
    "plain",
    "user_sim",
    "user_sim_fresh",
    "user_real_render",
    "fresh_draws",
)

SHARD_MAX_BYTES = 8_500_000  # < 9 MB non-LFS shard cap (upload-policy)


def derived_seed(*parts: object) -> int:
    """Deterministic per-unit seed derived from SEED-rooted string parts."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def english_majority(text: str) -> bool:
    """True when >50% of the LETTER characters are Latin-script (codepoint < 0x250).

    Rows with zero letters return False (cannot be verified English) -- a
    counted drop, never a silent keep.
    """
    import unicodedata

    letters = 0
    latin = 0
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            letters += 1
            if ord(ch) < 0x250:
                latin += 1
    if letters == 0:
        return False
    return latin / letters > 0.5


def text_digest(text: str) -> str:
    """Content-hygiene digest for logs/ledgers: sha256 prefix + length, never the text."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}|len={len(text)}"


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Text-mode JSONL iteration (never ``splitlines()`` -- U+2028/NEL safety)."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def atomic_write_json(path: Path, obj: object) -> None:
    """Write JSON atomically (tmp in the destination dir + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_metadata(extra: dict | None = None) -> dict:
    """Reproducibility metadata block (git provenance incl. dirty flag, env, ts)."""
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = {
        "issue": ISSUE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "argv": sys.argv,
        **as_metadata_dict(git_provenance()),
    }
    if extra:
        meta.update(extra)
    return meta


def progress(phase: str, k: int, n: int, key: str, t0: float) -> None:
    """Per-unit progress line (code-style checkpoint-per-phase contract)."""
    print(f"[{phase}] unit {k}/{n} {key} elapsed={time.time() - t0:.0f}s", flush=True)


class ShardWriter:
    """Rolling <9 MB JSONL shard writer: ``<stem>.shardNN.jsonl`` under ``out_dir``.

    Rows are written as single UTF-8 lines; ``close()`` finalizes the last
    shard and returns the shard paths. Writes are append-per-row so a crash
    loses at most the in-flight row (checkpoint-per-phase).
    """

    def __init__(self, out_dir: Path, stem: str, max_bytes: int = SHARD_MAX_BYTES):
        self.out_dir = out_dir
        self.stem = stem
        self.max_bytes = max_bytes
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._idx = 0
        self._bytes = 0
        self._rows = 0
        self._fh = None
        self.paths: list[Path] = []

    def _open_next(self) -> None:
        if self._fh is not None:
            self._fh.close()
        path = self.out_dir / f"{self.stem}.shard{self._idx:02d}.jsonl"
        # Resume-append friendliness: writing into an existing shard would
        # duplicate rows, so refuse -- resume logic must skip completed chunks.
        if path.exists() and path.stat().st_size > 0:
            raise RuntimeError(f"shard already exists (resume must skip completed chunks): {path}")
        self._fh = path.open("w", encoding="utf-8")
        self.paths.append(path)
        self._bytes = 0
        self._idx += 1

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        n = len(line.encode("utf-8"))
        if self._fh is None or self._bytes + n > self.max_bytes:
            self._open_next()
        assert self._fh is not None
        self._fh.write(line)
        self._fh.flush()
        self._bytes += n
        self._rows += 1

    def close(self) -> dict:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        return {"n_rows": self._rows, "shards": [str(p) for p in self.paths]}


def upload_stage_dir(local_dir: Path, prefix_rel: str) -> list[str]:
    """Fail-loud upload of a stage directory to the HF data repo + exact-set verify.

    One ``upload_folder`` commit (never a per-file loop -- #664 504-storms),
    then ``verify_repo_paths_uploaded`` over the expected file set. Returns the
    verified repo-relative paths. Raises on any failure.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    if not files:
        raise RuntimeError(f"upload_stage_dir: no files under {local_dir}")
    base_url = hub._upload(
        local_path=local_dir,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=prefix_rel,
        raise_on_error=True,
    )
    if not base_url:
        raise RuntimeError(f"upload returned no path for {prefix_rel} — durability loss, fail loud")
    expected = [f"{prefix_rel}/{p.relative_to(local_dir).as_posix()}" for p in files]
    missing = hub.verify_repo_paths_uploaded(
        HfApi(), HF_DATA_REPO, expected, path_in_repo=prefix_rel, repo_type="dataset"
    )
    if missing:
        raise RuntimeError(f"upload verify failed -- missing {len(missing)} paths: {missing[:5]}")
    print(f"[upload] {prefix_rel}: {len(expected)} files verified", flush=True)
    return expected


def stage_hf_prefix(prefix_rel: str, dest_root: Path, revision: str | None = None) -> Path:
    """Stage an HF data-repo prefix into ``dest_root`` (mirror root) and return
    the LOCAL directory holding the prefix contents (``dest_root/<prefix_rel>``).

    ``hub.stage_hub_prefix``'s dest is a MIRROR ROOT (#1774) -- files land at
    ``dest_root/<repo-relative path>``; this wrapper returns the mirrored leaf
    so callers cannot mis-nest.
    """
    from explore_persona_space.orchestrate import hub

    hub.stage_hub_prefix(
        HF_DATA_REPO, prefix_rel, dest_root, repo_type="dataset", revision=revision
    )
    leaf = dest_root / prefix_rel
    if not leaf.is_dir():
        raise RuntimeError(f"stage_hf_prefix: expected mirrored leaf missing: {leaf}")
    return leaf


class StageLedger:
    """Resumable per-stage chunk ledger (checkpoint-per-phase + regime-keyed resume).

    The ledger records a REGIME dict (every output-affecting parameter, built
    from generating parameters -- never hashed float arrays) plus the set of
    completed chunk keys. On resume with a mismatched regime it fails loud.
    """

    def __init__(self, path: Path, regime: dict):
        self.path = path
        self.regime = regime
        self.done: set[str] = set()
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("regime") != regime:
                raise RuntimeError(
                    f"StageLedger regime mismatch at {path}: on-disk "
                    f"{state.get('regime')} vs requested {regime} -- use a fresh out-root"
                )
            self.done = set(state.get("done", []))

    def is_done(self, key: str) -> bool:
        return key in self.done

    def mark_done(self, key: str) -> None:
        self.done.add(key)
        atomic_write_json(self.path, {"regime": self.regime, "done": sorted(self.done)})
