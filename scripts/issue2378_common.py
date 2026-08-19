"""Shared constants + helpers for issue #2378 (cross-framing context->answer transfer).

Consumed by ``scripts/issue2378_gen.py`` and ``scripts/issue2378_judge.py``
(and by the later capture/fits drivers). Everything here is deliberately
light-import: no torch/vllm/transformers at module top, so VM phases
(pool/bank building, judge waves) run on the repo venv while pod phases defer
their heavy imports into function bodies.

Key contracts (plan v6, tasks/running/2378/plans/plan.md):
- Cells: chat, plain_text, 5 story-Q cells, 4 dialogue cells, 2 user arms.
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
MINER_WINDOW_TOKENS = 250

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
DIALOG_CELLS: tuple[str, ...] = (
    "dialog_astra",
    "dialog_helios",
    "dialog_dana",
    "dialog_vex",
)  # no Wren dialogue cell (plan §5)
STORY_CELLS: tuple[str, ...] = STORY_Q_CELLS + DIALOG_CELLS
USER_CELLS: tuple[str, ...] = ("chat_user_real", "chat_user_sim")
ALL_CELLS: tuple[str, ...] = ("chat", "plain_text") + STORY_CELLS + USER_CELLS

_CHAR_BY_SLUG = {
    "astra": "Astra",
    "helios": "HELIOS",
    "wren": "Wren",
    "dana": "Dana",
    "vex": "Vex",
}
CELL_CHARACTER: dict[str, str] = {c: _CHAR_BY_SLUG[c.split("_", 1)[1]] for c in STORY_CELLS}
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
