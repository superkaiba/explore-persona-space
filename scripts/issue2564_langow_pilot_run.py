"""#2564 lang/oneword PILOT — pod driver: bank build + generation + capture + upload.

Two NEW minimal-pair axes appended to the frozen #2564 battery, as a SEPARATE
pilot bank (the frozen bank + its grid-completeness gates are untouched):

- ``answer_language`` (instruction axis): same English carrier question as the
  frozen bank, with a system-slot instruction naming the answer language.
  Values {english, chinese, spanish}; system strings EXACTLY
  "Answer in English." / "Answer in Chinese." / "Answer in Spanish.".
  Contexts: 12 carriers x (1 bare + 3 values) = 48. Pairs: install (value vs
  bare carrier, system == "") 36 + swap (all 3 value pairs) 36.
- ``query_content_oneword`` (query axis): 24 authored English question pairs
  (2 per carrier theme), each differing in EXACTLY ONE content word with
  similar general intent ("Should I adopt a dog?" vs "Should I adopt a cat?").
  Contexts: 24 x 2 = 48. Pairs: 24.

Phases (single process, sequential; ``--phase all`` default):

- ``gen``: K=10 draws per context, temp 1.0, seed_base 42, HF ``generate_batch``
  batch 16, max_new 2048 with per-cell cap-hit fraction + >2% whole-cell re-gen
  at 4096; ``.partial`` chunk-grain resume keyed on a regime fingerprint of the
  GENERATING PARAMETERS; per-cell atomic jsonl + done manifest; raw-completion
  upload (text persisted BEFORE capture, #779).
- ``capture``: teacher-forced v_A capture via the PINNED
  ``issue2162_run.capture_answer_states`` (span mean + tail-inclusive twin,
  fp16, layers 14/19/26, ``return_boundaries=True`` with the gate-4 EXACT
  boundary compare against the gen-side records) + v_C last-context-token
  capture (fp32); tensor upload.
- ``finalize``: terminal sentinel ``<out-root>/langow_done.json`` carrying
  per-phase counts + cap-hit fractions, written AFTER all uploads.

Frozen parent machinery is imported by PINNED BLOB (commit
``8265bcd75f78...`` on the issue-2564 branch) — ``bank2564.py`` (+ its values
JSON) and ``scripts/issue2162_run.py`` (the ``return_boundaries`` hunk is
branch-only) are extracted via ``git show`` into a tempdir and imported from
there; everything else resolves from the main-resident package tree. Bare
carriers are REGENERATED in-run (no cross-run tensor reuse).

Uploads: HF ``superkaiba1/explore-persona-space-data`` under
``issue2564_minpair/lang_oneword_pilot/{raw_completions,analysis_tensors,manifests}``.

Pod launch (fresh 1x H100, repo at main + fetched issue-2564 objects):

    uv run python scripts/issue2564_langow_pilot_run.py --phase all \
        --out-root /workspace/eps2564_langow --upload hf

Smoke blind-spot enumeration (``--tiny``):
- production model SUBSTITUTED: from-config 4-layer/64-hidden CPU model over
  the real vocab (``R.load_model_and_tokenizer`` tiny branch); the bf16 CUDA
  load and the production capture layers (14, 19, 26) never run under tiny
  (tiny captures layers (1, 2, 3)).
- ``model_revision`` UNRESOLVED under tiny ("unresolved-tiny") — the HfApi
  main->sha pin branch never runs.
- grid NARROWED: 2 contexts x 2 draws per cell (production 48 x 10); pairs
  filtered to surviving endpoints; ``max_new_tokens`` defaults to 64.
- cap-hit re-gen DEMOTED to an informational log line (production: whole-cell
  re-gen at 4096 when frac > 0.02).
- upload branch NOT exercised unless ``--upload hf`` is passed explicitly
  (tiny default ``--upload none``): ``upload_dir_sharded`` + the HF verify
  listing never run under the default smoke.
- ALWAYS-ON in both modes: render gates (exactly one "assistant" per render;
  empty-system prefix), ``changed_token_count >= 1`` per pair, the gate-4
  EXACT boundary compare (gen-side vs capture-side records), and the ctx
  re-tokenization drift assert.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch/numpy import — thread caps + credentials (code-style.md)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from explore_persona_space.atomic_io import (  # noqa: E402
    save_pt_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from explore_persona_space.experiments.issue1415.steering import generate_batch  # noqa: E402
from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: E402

logger = logging.getLogger("issue2564_langow")

REPO_ROOT = Path(__file__).resolve().parent.parent
assert (REPO_ROOT / "pyproject.toml").is_file(), REPO_ROOT

# ── pinned-blob imports (frozen parent machinery) ─────────────────────────

PIN = "8265bcd75f781d8e879e924de60063e536e58dcf"  # issue-2564 branch (frozen bank + MF-A capture)
PINNED_FILES = (
    "src/explore_persona_space/experiments/issue2564/bank2564.py",
    "src/explore_persona_space/experiments/issue2564/bank2564_values.json",
    "scripts/issue2162_run.py",
)


def _git_show(rel: str) -> bytes:
    """``git show PIN:rel`` with ONE fetch-and-retry (a fresh pod clone may not
    hold the issue-2564 branch objects yet). Fail-loud on the retry."""
    cmd = ["git", "show", f"{PIN}:{rel}"]
    out = subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ}, capture_output=True, check=False)
    if out.returncode != 0:
        logger.warning("[pin] %s missing locally — fetching origin issue-2564", PIN[:12])
        subprocess.run(
            ["git", "fetch", "origin", "issue-2564"],
            cwd=REPO_ROOT,
            env={**os.environ},
            check=True,
        )
        out = subprocess.run(
            cmd, cwd=REPO_ROOT, env={**os.environ}, capture_output=True, check=True
        )
    return out.stdout


def _import_pinned(name: str, path: Path):
    """Import an extracted pinned blob under a UNIQUE module name so main's
    same-named ``scripts/issue2162_run.py`` can never shadow it."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_PIN_DIR = Path(tempfile.mkdtemp(prefix="eps2564_langow_pin_"))
for _rel in PINNED_FILES:
    (_PIN_DIR / Path(_rel).name).write_bytes(_git_show(_rel))

BK = _import_pinned("bank2564_pinned", _PIN_DIR / "bank2564.py")
R = _import_pinned("issue2162_run_pinned", _PIN_DIR / "issue2162_run.py")

from explore_persona_space.analysis.extraction import (  # noqa: E402
    extract_layer_activations,
)

# ── constants ─────────────────────────────────────────────────────────────

ISSUE = 2564
HF_DATA_REPO = os.environ.get("EPM_2564_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")
HF_PREFIX = "issue2564_minpair/lang_oneword_pilot"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAP_LAYERS = (14, 19, 26)
TINY_LAYERS = (1, 2, 3)
HIDDEN = 3584
N_LAYERS = 28

ANCHOR_TEMPERATURE = 1.0
ANCHOR_MAX_NEW = 2048
REGEN_MAX_NEW = 4096
CAP_HIT_REGEN_FRAC = 0.02
CAP_HIT_BASIS = "retokenized_completion_len >= max_new_tokens"
ANCHOR_DRAWS = 10
GEN_BATCH = 16
CAPTURE_BATCH = 8

CELLS = ("answer_language", "query_content_oneword")

# Frozen pilot system strings (decision record: EXACT strings).
LANG_VALUES: dict[str, str] = {
    "english": "Answer in English.",
    "chinese": "Answer in Chinese.",
    "spanish": "Answer in Spanish.",
}

# One-word content swaps: 2 per carrier theme (24 pairs). Each pair differs in
# EXACTLY ONE content word with similar general intent; token matching is
# verified with the real tokenizer at bank build (changed_token_count recorded
# per pair, asserted >= 1). (pair_slug, carrier, question_a, question_b).
ONEWORD_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("p01", "c01", "Should I adopt a dog?", "Should I adopt a cat?"),
    ("p02", "c01", "What should I feed my dog?", "What should I feed my cat?"),
    ("p03", "c02", "Is it expensive to rent a home?", "Is it expensive to buy a home?"),
    ("p04", "c02", "Is it better to live in a house?", "Is it better to live in a condo?"),
    ("p05", "c03", "How should I spend a free weekend?", "How should I spend a free evening?"),
    ("p06", "c03", "What is a good weekend hobby?", "What is a good weekend activity?"),
    (
        "p07",
        "c04",
        "Should students be required to learn coding?",
        "Should students be required to learn statistics?",
    ),
    (
        "p08",
        "c04",
        "Is coding a useful skill for children?",
        "Is cooking a useful skill for children?",
    ),
    (
        "p09",
        "c05",
        "Is remote work better for productivity?",
        "Is remote work better for creativity?",
    ),
    ("p10", "c05", "Should companies allow remote work?", "Should schools allow remote work?"),
    (
        "p11",
        "c06",
        "What is the best way to meet new friends?",
        "What is the best way to meet new colleagues?",
    ),
    ("p12", "c06", "How can I make friends at work?", "How can I make friends at school?"),
    ("p13", "c07", "Should I read more fiction?", "Should I read more nonfiction?"),
    ("p14", "c07", "What is a good novel to read?", "What is a good biography to read?"),
    (
        "p15",
        "c08",
        "Is it worth traveling somewhere alone?",
        "Is it worth traveling somewhere abroad?",
    ),
    ("p16", "c08", "Should I travel by train?", "Should I travel by plane?"),
    ("p17", "c09", "How should a person choose a career?", "How should a person choose a hobby?"),
    (
        "p18",
        "c09",
        "What matters most when picking a job?",
        "What matters most when picking a city?",
    ),
    ("p19", "c10", "Is it better to save money?", "Is it better to invest money?"),
    (
        "p20",
        "c10",
        "How much money should I save each month?",
        "How much money should I spend each month?",
    ),
    ("p21", "c11", "Should someone follow a passion?", "Should someone follow a trend?"),
    (
        "p22",
        "c11",
        "Is a stable job more important than a passion?",
        "Is a stable income more important than a passion?",
    ),
    (
        "p23",
        "c12",
        "Is it better to exercise in the morning?",
        "Is it better to exercise in the evening?",
    ),
    ("p24", "c12", "Should I stretch before a run?", "Should I stretch before a swim?"),
)

PHASES = ("gen", "capture", "finalize")


# ── config ────────────────────────────────────────────────────────────────


@dataclass
class Cfg:
    """Duck-typed superset of the fields the pinned ``R.load_model_and_tokenizer``
    + ``R.capture_answer_states`` read (model_id / model_revision / tiny / hidden /
    n_layers / layers / device / capture_batch) plus this driver's own knobs."""

    phase: str
    out_root: Path
    model_id: str
    model_revision: str
    tiny: bool
    hidden: int
    n_layers: int
    layers: tuple[int, ...]
    device: str
    capture_batch: int
    gen_batch: int
    draws: int
    max_new_tokens: int
    seed_base: int
    upload: str  # "hf" | "none"

    @property
    def anchors_dir(self) -> Path:
        return self.out_root / "anchors"

    @property
    def va_dir(self) -> Path:
        return self.out_root / "va_store"

    @property
    def vc_dir(self) -> Path:
        return self.out_root / "vc_store"

    @property
    def manifest_dir(self) -> Path:
        return self.out_root / "manifests"

    @property
    def quarantine_dir(self) -> Path:
        return self.out_root / "quarantine"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=(*PHASES, "all"), default="all")
    ap.add_argument("--out-root", default="/workspace/eps2564_langow")
    ap.add_argument(
        "--upload",
        choices=("hf", "none"),
        default=None,
        help="default: hf in production, none under --tiny",
    )
    ap.add_argument(
        "--tiny", action="store_true", help="CPU smoke: tiny model, 2 contexts x 2 draws per cell"
    )
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--gen-batch", type=int, default=GEN_BATCH)
    ap.add_argument("--capture-batch", type=int, default=CAPTURE_BATCH)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="verify argparse attrs + pinned-call signatures, then exit 0",
    )
    return ap


def _resolve_model_revision(tiny: bool) -> str:
    """Pin main -> resolved sha ONCE per run (#2061); fail loud outside tiny mode."""
    if tiny:
        return "unresolved-tiny"
    from huggingface_hub import HfApi

    sha = HfApi().model_info(MODEL_ID).sha
    assert sha, f"could not resolve model revision for {MODEL_ID}"
    return sha


def build_cfg(args: argparse.Namespace) -> Cfg:
    device = args.device or ("cpu" if args.tiny else "cuda:0")
    return Cfg(
        phase=args.phase,
        out_root=Path(args.out_root),
        model_id=MODEL_ID,
        model_revision=_resolve_model_revision(args.tiny),
        tiny=bool(args.tiny),
        hidden=64 if args.tiny else HIDDEN,
        n_layers=4 if args.tiny else N_LAYERS,
        layers=TINY_LAYERS if args.tiny else MAP_LAYERS,
        device=device,
        capture_batch=args.capture_batch,
        gen_batch=args.gen_batch,
        draws=args.draws if args.draws is not None else (2 if args.tiny else ANCHOR_DRAWS),
        max_new_tokens=(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else (64 if args.tiny else ANCHOR_MAX_NEW)
        ),
        seed_base=args.seed_base,
        upload=args.upload if args.upload is not None else ("none" if args.tiny else "hf"),
    )


def _import_check() -> None:
    """Module-level (never in ``main`` — the local-import shadowing trap): argparse
    attribute completeness + signature binds on the pinned call surface."""
    import inspect

    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    for fn, needed in (
        (
            R.capture_answer_states,
            {"payloads", "positions", "tail_inclusive", "return_boundaries"},
        ),
        (
            generate_batch,
            {"n", "hook", "max_new_tokens", "temperature", "seed_base", "render_fn", "ids_fn"},
        ),
    ):
        params = set(inspect.signature(fn).parameters)
        missing = needed - params
        assert not missing, (fn.__name__, sorted(missing))
    for name in (
        "render_context",
        "context_token_ids",
        "changed_token_count",
        "load_values",
        "context_id",
        "pair_id",
    ):
        assert callable(getattr(BK, name)), name
    for name in ("load_model_and_tokenizer", "eot_tail_ids", "cap_hit", "_right_pad"):
        assert callable(getattr(R, name)), name
    print("[import-check] ok: pinned modules + call signatures resolve", flush=True)


# ── small io / provenance helpers ─────────────────────────────────────────


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: Path, tolerate_torn_tail: bool = False) -> list[dict]:
    """Text-mode line iteration (never ``splitlines()`` — U+2028 shred gotcha);
    a torn final line is dropped only under ``tolerate_torn_tail``."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        lines = [ln for ln in fh.read().split("\n") if ln.strip()]
    for k, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if tolerate_torn_tail and k == len(lines) - 1:
                logger.warning("[resume] dropping torn tail line of %s", path.name)
                return rows
            raise
    return rows


_REPRO_CACHE: dict | None = None


def _repro(cfg: Cfg, phase: str) -> dict:
    """Reproducibility metadata carried by every persisted artifact."""
    global _REPRO_CACHE
    if _REPRO_CACHE is None:
        import transformers

        from explore_persona_space.orchestrate.provenance import (
            as_metadata_dict,
            git_provenance,
        )

        _REPRO_CACHE = {
            **as_metadata_dict(git_provenance()),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "pin": PIN,
        }
    return {
        **_REPRO_CACHE,
        "phase": phase,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "tiny": cfg.tiny,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def _sha16(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _regime_fp(cfg: Cfg, extra: dict | None = None) -> str:
    """16-hex fingerprint of the GENERATING PARAMETERS (never recomputed floats,
    #1336) — the resume / done-manifest key."""
    base = {
        "issue": ISSUE,
        "pin": PIN,
        "model_id": cfg.model_id,
        "model_revision": cfg.model_revision,
        "tiny": cfg.tiny,
        "draws": cfg.draws,
        "gen_batch": cfg.gen_batch,
        "seed_base": cfg.seed_base,
        "temperature": str(ANCHOR_TEMPERATURE),
        "max_new_tokens": cfg.max_new_tokens,
        "lang_values": LANG_VALUES,
        "oneword_sha": _sha16(list(ONEWORD_PAIRS)),
        # Upload mode is part of the resume key: a --upload none run must never
        # satisfy a later --upload hf run's phase sentinel (review finding 1).
        "upload": cfg.upload,
    }
    if extra:
        base.update(extra)
    return _sha16(base)


def _cell_fp(cfg: Cfg, phase: str, cell: str) -> str:
    return _regime_fp(cfg, {"phase": phase, "cell": cell})


# ── pilot bank ────────────────────────────────────────────────────────────


def build_pilot_bank(cfg: Cfg, tok) -> dict:
    """Pilot contexts + pairs for the two new axes, gated:

    - render gate: exactly one "assistant" occurrence per render; empty-system
      renders start with the frozen empty-system prefix;
    - ``changed_token_count`` >= 1 per pair over FULL rendered-prompt token ids
      (recorded per pair);
    - counts: 48 + 48 contexts, 36 install + 36 swap + 24 oneword pairs
      (production; the tiny slice keeps pairs whose BOTH endpoints survive).
    """
    values = BK.load_values()
    carriers = values["carriers"]
    contexts: dict[str, dict] = {}
    per_cell: dict[str, list[str]] = {c: [] for c in CELLS}

    def _add(ctx: dict) -> None:
        assert ctx["id"] not in contexts, ctx["id"]
        contexts[ctx["id"]] = ctx
        per_cell[ctx["cell"]].append(ctx["id"])

    for carrier in BK.CARRIER_IDS:
        car = carriers[carrier]
        _add(
            {
                "id": BK.context_id("answer_language", "bare", carrier),
                "cell": "answer_language",
                "kind": "bare",
                "value_id": "bare",
                "carrier": carrier,
                "form": "question",
                "system": "",
                "user": car["question"],
            }
        )
        for lang, system in LANG_VALUES.items():
            _add(
                {
                    "id": BK.context_id("answer_language", lang, carrier),
                    "cell": "answer_language",
                    "kind": "value",
                    "value_id": lang,
                    "carrier": carrier,
                    "form": "question",
                    "system": system,
                    "user": car["question"],
                }
            )
    for slug, carrier, q_a, q_b in ONEWORD_PAIRS:
        for side, q in (("a", q_a), ("b", q_b)):
            _add(
                {
                    "id": BK.context_id("query_content_oneword", f"{slug}{side}", carrier),
                    "cell": "query_content_oneword",
                    "kind": "E",
                    "value_id": f"{slug}{side}",
                    "carrier": carrier,
                    "form": "question",
                    "system": "",
                    "user": q,
                }
            )
    assert len(per_cell["answer_language"]) == 48, len(per_cell["answer_language"])
    assert len(per_cell["query_content_oneword"]) == 48, len(per_cell["query_content_oneword"])

    pairs: list[dict] = []
    langs = tuple(LANG_VALUES)
    for carrier in BK.CARRIER_IDS:
        bare = BK.context_id("answer_language", "bare", carrier)
        for lang in langs:
            pairs.append(
                {
                    "pair_id": BK.pair_id("install", "answer_language", lang, "bare", carrier),
                    "pair_class": "install",
                    "axis": "answer_language",
                    "carrier": carrier,
                    "value_a": lang,
                    "value_b": "bare",
                    "a": BK.context_id("answer_language", lang, carrier),
                    "b": bare,
                }
            )
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                va, vb = langs[i], langs[j]
                pairs.append(
                    {
                        "pair_id": BK.pair_id("swap", "answer_language", va, vb, carrier),
                        "pair_class": "swap",
                        "axis": "answer_language",
                        "carrier": carrier,
                        "value_a": va,
                        "value_b": vb,
                        "a": BK.context_id("answer_language", va, carrier),
                        "b": BK.context_id("answer_language", vb, carrier),
                    }
                )
    for slug, carrier, _q_a, _q_b in ONEWORD_PAIRS:
        pairs.append(
            {
                "pair_id": BK.pair_id(
                    "query_content_oneword",
                    "query_content_oneword",
                    f"{slug}a",
                    f"{slug}b",
                    carrier,
                ),
                "pair_class": "query_content_oneword",
                "axis": "query_content_oneword",
                "carrier": carrier,
                "value_a": f"{slug}a",
                "value_b": f"{slug}b",
                "a": BK.context_id("query_content_oneword", f"{slug}a", carrier),
                "b": BK.context_id("query_content_oneword", f"{slug}b", carrier),
            }
        )
    n_by_class = {
        cls: sum(1 for p in pairs if p["pair_class"] == cls)
        for cls in ("install", "swap", "query_content_oneword")
    }
    assert n_by_class == {"install": 36, "swap": 36, "query_content_oneword": 24}, n_by_class

    if cfg.tiny:
        keep_ids = {cid for cell in CELLS for cid in per_cell[cell][:2]}
        contexts = {cid: c for cid, c in contexts.items() if cid in keep_ids}
        pairs = [p for p in pairs if p["a"] in contexts and p["b"] in contexts]
        per_cell = {cell: [cid for cid in ids if cid in contexts] for cell, ids in per_cell.items()}
        assert pairs, "tiny slice kept no pair — context ordering broke the pair-survival invariant"

    ids_by_ctx: dict[str, list[int]] = {}
    for cid, ctx in contexts.items():
        rendered = BK.render_context(tok, ctx)
        assert rendered.count("assistant") == 1, (cid, rendered.count("assistant"))
        if ctx["system"] == "":
            assert rendered.startswith("<|im_start|>system\n<|im_end|>\n"), cid
        ids_by_ctx[cid] = BK.context_token_ids(tok, ctx)
    for p in pairs:
        chg = BK.changed_token_count(ids_by_ctx[p["a"]], ids_by_ctx[p["b"]])
        assert chg >= 1, (p["pair_id"], "identical rendered prompts")
        p["changed_tokens"] = int(chg)

    return {"contexts": contexts, "pairs": pairs, "per_cell": per_cell}


def write_bank_manifest(cfg: Cfg, bank: dict) -> None:
    write_json_atomic(
        cfg.manifest_dir / "pilot_bank.json",
        {
            "issue": ISSUE,
            "regime_fp": _regime_fp(cfg, {"phase": "bank"}),
            "contexts": list(bank["contexts"].values()),
            "pairs": bank["pairs"],
            "n_contexts": len(bank["contexts"]),
            "n_pairs": len(bank["pairs"]),
            "lang_values": LANG_VALUES,
            "repro": _repro(cfg, "bank"),
        },
    )


# ── phase: gen ────────────────────────────────────────────────────────────


def _gen_row(
    cfg: Cfg,
    ctx: dict,
    ctx_len: int,
    n_eot: int,
    draw: int,
    chunk: int,
    text: str,
    n_comp: int,
    max_new: int,
) -> dict:
    """Generation-side per-row record incl. the gate-4 span fields (compared
    EXACTLY against the capture path's own ``boundaries`` records)."""
    return {
        "context_id": ctx["id"],
        "cell": ctx["cell"],
        "kind": ctx["kind"],
        "value_id": ctx["value_id"],
        "carrier": ctx["carrier"],
        "form": ctx["form"],
        "draw": draw,
        "seed": cfg.seed_base + draw,
        "chunk": chunk,
        "temperature": ANCHOR_TEMPERATURE,
        "max_new_tokens": max_new,
        "ctx_len": ctx_len,
        "n_completion_tokens_gen": n_comp,
        "span_start": ctx_len,
        "span_end": ctx_len + n_comp,
        "tail_end": ctx_len + n_comp + n_eot,
        "cap_hit": R.cap_hit(n_comp, max_new),
        "cap_hit_basis": CAP_HIT_BASIS,
        "text": text,
    }


def _generate_cell(
    cfg: Cfg, model, tok, eot_ids: list[int], cell: str, ctxs: list[dict], max_new: int
) -> list[dict]:
    """All draws for one cell, chunk-grain checkpointed to a ``.partial`` sidecar
    (fp-header keyed; quarantined on mismatch; torn-tail-tolerant resume)."""
    part = cfg.anchors_dir / f"anchors_{cell}.max{max_new}.partial"
    part_fp = _regime_fp(cfg, {"phase": "gen", "cell": cell, "max_new_call": max_new})
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    chunks = [ctxs[i : i + cfg.gen_batch] for i in range(0, len(ctxs), cfg.gen_batch)]
    prior: list[dict] = []
    if part.is_file():
        raw = _read_jsonl(part, tolerate_torn_tail=True)
        header = raw[0] if raw else None
        if (
            header is not None
            and header.get("partial_header")
            and header.get("regime_fp") == part_fp
        ):
            prior = raw[1:]
        else:
            cfg.quarantine_dir.mkdir(parents=True, exist_ok=True)
            dest = cfg.quarantine_dir / f"{time.time_ns()}.{part.name}"
            os.replace(part, dest)
            logger.warning("[gen:%s] quarantined stale .partial -> %s", cell, dest)
    if not part.is_file():
        with part.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"partial_header": 1, "regime_fp": part_fp}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    by_chunk: dict[int, list[dict]] = {}
    for r in prior:
        by_chunk.setdefault(int(r["chunk"]), []).append(r)
    complete = {
        ci
        for ci, rs in by_chunk.items()
        if ci < len(chunks)
        and len(rs) == len(chunks[ci]) * cfg.draws
        and {r["context_id"] for r in rs} == {c["id"] for c in chunks[ci]}
    }
    rows: list[dict] = [r for ci in sorted(complete) for r in by_chunk[ci]]
    if complete:
        logger.info("[gen:%s] resumed %d/%d chunks", cell, len(complete), len(chunks))
    n_eot = len(eot_ids)
    t_cell = time.time()
    for ci, chunk in enumerate(chunks):
        if ci in complete:
            continue
        t0 = time.time()
        results = generate_batch(
            model,
            tok,
            chunk,
            n=cfg.draws,
            hook=None,
            max_new_tokens=max_new,
            temperature=ANCHOR_TEMPERATURE,
            seed_base=cfg.seed_base,
            render_fn=BK.render_context,
            ids_fn=BK.context_token_ids,
        )
        wall = time.time() - t0
        new_rows: list[dict] = []
        for b, ctx in enumerate(chunk):
            ctx_len = len(BK.context_token_ids(tok, ctx))
            for i in range(cfg.draws):
                text = results[b][i]
                n_comp = len(tok(text, add_special_tokens=False)["input_ids"])
                new_rows.append(_gen_row(cfg, ctx, ctx_len, n_eot, i, ci, text, n_comp, max_new))
        with part.open("a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new_rows))
            fh.flush()
            os.fsync(fh.fileno())
        rows.extend(new_rows)
        print(
            f"[gen:{cell}] unit {ci + 1}/{len(chunks)} rows={len(new_rows)} "
            f"elapsed={time.time() - t_cell:.1f}s chunk_wall={wall:.1f}s",
            flush=True,
        )
    rows.sort(key=lambda r: (r["chunk"], r["context_id"], r["draw"]))
    return rows


def _gen_cell_complete(cfg: Cfg, cell: str) -> bool:
    m = _read_json(cfg.manifest_dir / f"anchors_{cell}.done.json")
    return (
        m is not None
        and m.get("regime_fp") == _cell_fp(cfg, "gen", cell)
        and (cfg.anchors_dir / f"anchors_{cell}.jsonl").is_file()
    )


def _gen_cell(cfg: Cfg, model, tok, eot_ids: list[int], cell: str, ctxs: list[dict]) -> None:
    """One cell: generate -> cap-hit check (>2% => whole-cell re-gen at 4096;
    DEMOTED to informational under --tiny) -> atomic final jsonl + done manifest."""
    out_path = cfg.anchors_dir / f"anchors_{cell}.jsonl"
    rows = _generate_cell(cfg, model, tok, eot_ids, cell, ctxs, cfg.max_new_tokens)
    frac = sum(1 for r in rows if r["cap_hit"]) / max(1, len(rows))
    regen_frac = None
    max_new_final = cfg.max_new_tokens
    if frac > CAP_HIT_REGEN_FRAC and cfg.max_new_tokens < REGEN_MAX_NEW:
        if cfg.tiny:
            logger.info(
                "[gen:%s] tiny: cap-hit frac %.4f > %.2f (informational — no re-gen)",
                cell,
                frac,
                CAP_HIT_REGEN_FRAC,
            )
        else:
            logger.warning(
                "[gen:%s] cap-hit frac %.4f > %.2f — re-gen at max_new=%d",
                cell,
                frac,
                CAP_HIT_REGEN_FRAC,
                REGEN_MAX_NEW,
            )
            write_jsonl_atomic(
                cfg.anchors_dir / f"anchors_{cell}.capped{cfg.max_new_tokens}.jsonl", rows
            )
            rows = _generate_cell(cfg, model, tok, eot_ids, cell, ctxs, REGEN_MAX_NEW)
            regen_frac = sum(1 for r in rows if r["cap_hit"]) / max(1, len(rows))
            max_new_final = REGEN_MAX_NEW
    write_jsonl_atomic(out_path, rows)
    for max_new in (cfg.max_new_tokens, REGEN_MAX_NEW):
        p = cfg.anchors_dir / f"anchors_{cell}.max{max_new}.partial"
        if p.is_file():
            p.unlink()
    write_json_atomic(
        cfg.manifest_dir / f"anchors_{cell}.done.json",
        {
            "cell": cell,
            "regime_fp": _cell_fp(cfg, "gen", cell),
            "n_contexts": len(ctxs),
            "n_rows": len(rows),
            "cap_hit_frac": frac,
            "cap_hit_frac_regen": regen_frac,
            "max_new_tokens_final": max_new_final,
            "repro": _repro(cfg, "gen"),
        },
    )


def _upload_summary(res) -> dict:
    return {
        "repo_id": res.repo_id,
        "uploaded": len(res.uploaded),
        "rerouted": len(res.rerouted),
        "skipped_existing": len(res.skipped_existing),
    }


def phase_gen(cfg: Cfg, bank: dict, model, tok) -> int:
    print("[phase=gen] start", flush=True)
    eot_ids = R.eot_tail_ids(tok)
    write_bank_manifest(cfg, bank)
    sentinel = cfg.out_root / "langow_gen_done.json"
    pending = [c for c in CELLS if not _gen_cell_complete(cfg, c)]
    s = _read_json(sentinel)
    if not pending and s is not None and s.get("regime_fp") == _regime_fp(cfg, {"phase": "gen"}):
        logger.info("[gen] all cells complete + sentinel present — skipping")
        return 0
    for cell in CELLS:
        if _gen_cell_complete(cfg, cell):
            logger.info("[gen:%s] done manifest present — skipping", cell)
            continue
        ctxs = [bank["contexts"][cid] for cid in bank["per_cell"][cell]]
        _gen_cell(cfg, model, tok, eot_ids, cell, ctxs)
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        # Rollout TEXT persists to HF BEFORE any capture reduce (#779).
        res = upload_dir_sharded(
            cfg.anchors_dir,
            HF_DATA_REPO,
            f"{HF_PREFIX}/raw_completions/anchors",
            shard_glob="*.jsonl",
            resume_skip=False,
            delete_local=False,
        )
        upload["anchors"] = _upload_summary(res)
    write_json_atomic(
        sentinel,
        {
            "regime_fp": _regime_fp(cfg, {"phase": "gen"}),
            "cells": {c: _read_json(cfg.manifest_dir / f"anchors_{c}.done.json") for c in CELLS},
            "upload": upload,
            "repro": _repro(cfg, "gen"),
        },
    )
    print("[phase=gen] sentinel written", flush=True)
    return 0


# ── phase: capture ────────────────────────────────────────────────────────


def _capture_vc(cfg: Cfg, model, tok, contexts: list[dict]) -> None:
    """v_C context-end capture: last real context token, all pilot contexts, fp32."""
    layers = list(cfg.layers)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids_all = [BK.context_token_ids(tok, c) for c in contexts]
    vc = torch.zeros(len(contexts), len(layers), cfg.hidden, dtype=torch.float32)
    t0 = time.time()
    n_chunks = (len(contexts) + cfg.capture_batch - 1) // cfg.capture_batch
    with torch.no_grad():
        for ci in range(n_chunks):
            lo = ci * cfg.capture_batch
            chunk_ids = ids_all[lo : lo + cfg.capture_batch]
            ids, mask = R._right_pad(chunk_ids, pad_id, cfg.device)
            acts = extract_layer_activations(model, ids, layers, attention_mask=mask)
            for b, row_ids in enumerate(chunk_ids):
                pos = len(row_ids) - 1
                for li, layer in enumerate(layers):
                    vc[lo + b, li] = acts[layer][b, pos].float().cpu()
            print(f"[vc] unit {ci + 1}/{n_chunks} elapsed={time.time() - t0:.1f}s", flush=True)
    assert vc.shape == (len(contexts), len(layers), cfg.hidden), vc.shape
    save_pt_atomic(
        cfg.vc_dir / "vc_langow_bank.pt",
        {
            "issue": ISSUE,
            "layers": layers,
            "context_ids": [c["id"] for c in contexts],
            "vc": vc,
            "dtype": "fp32",
            "position": "context_end_last_token",
            "repro": _repro(cfg, "capture"),
        },
    )
    write_json_atomic(
        cfg.manifest_dir / "vc_langow.done.json",
        {
            "regime_fp": _regime_fp(cfg, {"phase": "capture", "leg": "vc"}),
            "n_contexts": len(contexts),
            "repro": _repro(cfg, "capture"),
        },
    )


def _capture_cell_va(cfg: Cfg, model, tok, eot_ids: list[int], cell: str, ctx_by_id: dict) -> None:
    """One cell: teacher-forced v_A capture (span mean + tail twin, fp16) with the
    pinned ``return_boundaries`` records, EXACT-compared against gen-side records."""
    rows = _read_jsonl(cfg.anchors_dir / f"anchors_{cell}.jsonl")
    assert rows, f"no anchor rows for cell {cell}"
    ctx_ids_by_row = [BK.context_token_ids(tok, ctx_by_id[r["context_id"]]) for r in rows]
    for r, ids in zip(rows, ctx_ids_by_row):
        assert len(ids) == r["ctx_len"], (
            f"ctx re-tokenization drift for {r['context_id']}: {len(ids)} != {r['ctx_len']}"
        )
    completions = [r["text"] for r in rows]
    t0 = time.time()
    out = R.capture_answer_states(
        cfg,
        model,
        tok,
        ctx_ids_by_row,
        completions,
        eot_ids,
        tail_inclusive=True,
        return_boundaries=True,
    )
    wall = time.time() - t0
    bounds = out["boundaries"]
    assert len(bounds) == len(rows), (len(bounds), len(rows))
    for r, b, n in zip(rows, bounds, out["n_completion_tokens"]):
        assert int(n) == r["n_completion_tokens_gen"], (
            f"completion-len drift {cell}/{r['context_id']}/d{r['draw']}: "
            f"capture {n} != gen {r['n_completion_tokens_gen']}"
        )
        for key in ("ctx_len", "span_start", "span_end", "tail_end"):
            assert b[key] == r[key], (
                f"gate-4 boundary mismatch {cell}/{r['context_id']}/d{r['draw']} "
                f"{key}: capture {b[key]} != gen {r[key]}"
            )
    index = [
        {"context_id": r["context_id"], "cell": r["cell"], "draw": r["draw"], **b}
        for r, b in zip(rows, bounds)
    ]
    save_pt_atomic(
        cfg.va_dir / f"va_langow_{cell}.pt",
        {
            "issue": ISSUE,
            "cell": cell,
            "layers": list(cfg.layers),
            "index": index,
            "va_span": out["va_span"],
            "va_tail_incl": out["va_tail_incl"],
            "poolings": ["span_mean", "tail_inclusive_mean"],
            "empty_rows": out["empty_rows"],
            "eot_ids": eot_ids,
            "max_new_tokens": rows[0]["max_new_tokens"] if rows else None,
            "repro": _repro(cfg, "capture"),
        },
    )
    write_json_atomic(
        cfg.manifest_dir / f"va_langow_{cell}.done.json",
        {
            "cell": cell,
            "regime_fp": _cell_fp(cfg, "capture", cell),
            "n_rows": len(rows),
            "n_empty_rows": len(out["empty_rows"]),
            "per_row_s": wall / max(1, len(rows)),
            "repro": _repro(cfg, "capture"),
        },
    )


def _va_cell_complete(cfg: Cfg, cell: str) -> bool:
    m = _read_json(cfg.manifest_dir / f"va_langow_{cell}.done.json")
    return (
        m is not None
        and m.get("regime_fp") == _cell_fp(cfg, "capture", cell)
        and (cfg.va_dir / f"va_langow_{cell}.pt").is_file()
    )


def _vc_complete(cfg: Cfg) -> bool:
    m = _read_json(cfg.manifest_dir / "vc_langow.done.json")
    return (
        m is not None
        and m.get("regime_fp") == _regime_fp(cfg, {"phase": "capture", "leg": "vc"})
        and (cfg.vc_dir / "vc_langow_bank.pt").is_file()
    )


def phase_capture(cfg: Cfg, bank: dict, model, tok) -> int:
    print("[phase=capture] start", flush=True)
    eot_ids = R.eot_tail_ids(tok)
    sentinel = cfg.out_root / "langow_capture_done.json"
    contexts = [bank["contexts"][cid] for cell in CELLS for cid in bank["per_cell"][cell]]
    pending_va = [c for c in CELLS if not _va_cell_complete(cfg, c)]
    s = _read_json(sentinel)
    if (
        not pending_va
        and _vc_complete(cfg)
        and s is not None
        and s.get("regime_fp") == _regime_fp(cfg, {"phase": "capture"})
    ):
        logger.info("[capture] all cells + vc complete + sentinel — skipping")
        return 0
    if not _vc_complete(cfg):
        _capture_vc(cfg, model, tok, contexts)
    ctx_by_id = bank["contexts"]
    for cell in CELLS:
        if _va_cell_complete(cfg, cell):
            logger.info("[capture:%s] done manifest present — skipping", cell)
            continue
        _capture_cell_va(cfg, model, tok, eot_ids, cell, ctx_by_id)
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        for name, local_dir, glob in (
            ("va", cfg.va_dir, "*.pt"),
            ("vc", cfg.vc_dir, "*.pt"),
            ("manifests", cfg.manifest_dir, "*.json"),
        ):
            res = upload_dir_sharded(
                local_dir,
                HF_DATA_REPO,
                f"{HF_PREFIX}/analysis_tensors/{name}"
                if name != "manifests"
                else f"{HF_PREFIX}/manifests",
                shard_glob=glob,
                resume_skip=False,
                delete_local=False,
            )
            upload[name] = _upload_summary(res)
    write_json_atomic(
        sentinel,
        {
            "regime_fp": _regime_fp(cfg, {"phase": "capture"}),
            "n_contexts_vc": len(contexts),
            "cells": {c: _read_json(cfg.manifest_dir / f"va_langow_{c}.done.json") for c in CELLS},
            "upload": upload,
            "repro": _repro(cfg, "capture"),
        },
    )
    print("[phase=capture] sentinel written", flush=True)
    return 0


# ── phase: finalize ───────────────────────────────────────────────────────


def phase_finalize(cfg: Cfg) -> int:
    """Terminal sentinel — written LAST, after all uploads (upload-policy)."""
    print("[phase=finalize] start", flush=True)
    gen_s = _read_json(cfg.out_root / "langow_gen_done.json")
    cap_s = _read_json(cfg.out_root / "langow_capture_done.json")
    assert gen_s is not None, "gen sentinel missing — run --phase gen first"
    assert cap_s is not None, "capture sentinel missing — run --phase capture first"
    per_cell = {}
    for cell in CELLS:
        g = _read_json(cfg.manifest_dir / f"anchors_{cell}.done.json") or {}
        v = _read_json(cfg.manifest_dir / f"va_langow_{cell}.done.json") or {}
        per_cell[cell] = {
            "n_contexts": g.get("n_contexts"),
            "n_rows_gen": g.get("n_rows"),
            "cap_hit_frac": g.get("cap_hit_frac"),
            "cap_hit_frac_regen": g.get("cap_hit_frac_regen"),
            "max_new_tokens_final": g.get("max_new_tokens_final"),
            "n_rows_captured": v.get("n_rows"),
            "n_empty_rows": v.get("n_empty_rows"),
        }
    write_json_atomic(
        cfg.out_root / "langow_done.json",
        {
            "issue": ISSUE,
            "status": "done",
            "regime_fp": _regime_fp(cfg, {"phase": "finalize"}),
            "cells": per_cell,
            "upload_gen": gen_s.get("upload"),
            "upload_capture": cap_s.get("upload"),
            "hf_prefix": HF_PREFIX,
            "repro": _repro(cfg, "finalize"),
        },
    )
    print("[phase=done] langow_done.json written", flush=True)
    return 0


# ── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.import_check:
        _import_check()
        return 0
    cfg = build_cfg(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    model = tok = None
    bank = None
    if any(p in ("gen", "capture") for p in phases):
        model, tok = R.load_model_and_tokenizer(cfg)
        bank = build_pilot_bank(cfg, tok)
        print(
            f"[bank] {len(bank['contexts'])} contexts / {len(bank['pairs'])} pairs "
            f"(tiny={cfg.tiny})",
            flush=True,
        )
    rc = 0
    for phase in phases:
        if phase == "gen":
            rc = phase_gen(cfg, bank, model, tok)
        elif phase == "capture":
            rc = phase_capture(cfg, bank, model, tok)
        elif phase == "finalize":
            rc = phase_finalize(cfg)
        if rc != 0:
            return rc
    return rc


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
