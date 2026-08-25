"""#2544 generation + activation-capture driver (Olmo-3-7B 15-rung stage map).

Phases (subprocess-per-phase; the #1902 dispatcher contracts):

- ``--phase config``    P0: budget re-check, exemplar bank (M3), pinned
  subsets, ladder validation; uploads to ``issue2544_stage_map/config/``.
- ``--phase stage``     pod-side staging: corpus + config down, stage
  manifest (corpus/config hashes recorded ONCE — M2 hash-at-write inputs).
- ``--phase pilot``     P1 (``--init`` pins + tokenizer/config battery;
  ``--worker`` pilot units; ``--finalize`` Gate A + report).
- ``--phase pass1``     P2+P3a (``--init`` queue; ``--worker`` gen+capture
  units under the K=4 rung-residency admission (M1); ``--finalize`` = P2.5
  diagnostics + shared intersection + Gate A').
- ``--phase pass2``     P3b (band B6 = B5 ∪ {l_FA} captures; needs the P4a
  layer-freeze record ``fits/layer_freeze.json`` — Unit B's output).
- ``--phase fits``      registration point -> ``issue2544_fits.py`` (Unit B).

Contracts inherited from #1902: ``/workspace/logs/issue-2544-*.json`` phase
sentinels, ``[phase=...]`` breadcrumbs, per-unit persistence + resume,
explicit ``sys.exit(0)``, ``VLLM_WORKER_MULTIPROC_METHOD=spawn`` at module
top. IMPORT ORDER: ``issue2544_common`` first (it sets the ladder + write
prefix env BEFORE ``issue1902_common`` binds its constants).

Smoke: ``--smoke`` slices rows/subsets (scale dials only) and REFUSES to run
against the production HF write prefix — the smoke runner must export
``EPM_ISSUE1902_HF_WRITE_PREFIX=issue2544_stage_map/_smoke`` (and usually
``EPM_ISSUE1902_SMOKE_MODEL_DIR``). Smoke-downgraded gates (Gate A/A' floors,
Olmo-3 absolute config asserts) log ``[smoke-downgrade]`` lines — the Unit C
blind-spot enumeration reads them.

Content hygiene: LMSYS text — no corpus/exemplar/rollout row text is ever
printed or logged; digests are ids + counts + hashes only.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2544_common as C2  # noqa: E402  (MUST precede issue1902_run — env-ordered)
import issue1902_common as C  # noqa: E402
import issue1902_run as R  # noqa: E402

logger = logging.getLogger("issue2544_run")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

ISSUE = C2.ISSUE
RECIPE_VERSION = C2.RECIPE_VERSION

# §9 booked walls (the P1 cost-re-projection gate references; plan §9 table).
BOOKED_PASS_WALL_H = 8.5 + 3.5  # P2+P3a + P3b planned_wall_h
CAPTURE_LAYER_FULL = "full17"
SMOKE_SUBSET_SIZES: dict[str, int] = {"pilot": 8, "reliability": 4, "robust": 8, "natgen": 4}
SMOKE_EXEMPLAR_QUOTAS: dict[str, int] = {"generic": 40, "math": 12, "code": 12}
# MEASURED (Unit C smoke, 2026-08-24): the production quotas {120,40,40} fill at
# 34,834 scanned LMSYS rows under the production eligibility filters (math is
# the scarce class); 6,000 left the math class at ~3 rows and the bank build
# raised "spares infeasible". The stream stops EARLY once the smoke quotas
# {40,12,12} fill, so this bound only pays when classes are rare.
SMOKE_EXEMPLAR_SCAN_CAP = 40_000
WILSON_Z = 1.959963984540054  # 95% two-sided


# ── small shared helpers ─────────────────────────────────────────────────────


def _metadata() -> dict[str, Any]:
    """Reproducibility metadata (issue 2544; reuses R's git-sha degradation)."""
    md = R._metadata()
    md["issue"] = ISSUE
    md["recipe_version"] = RECIPE_VERSION
    md["code_sha"] = C2.code_sha()
    return md


def write_sentinel(out_root: Path, phase: str, note: dict[str, Any], *, smoke: bool) -> Path:
    """issue-2544 phase-done sentinel (poll_pipeline drain contract)."""
    sdir = R._sentinel_dir(out_root)
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"issue-{ISSUE}-{phase}-done-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "kind": "epm:progress",
        "version": 1,
        "task_id": ISSUE,
        "by": "issue2544_run",
        "ts": R._now_iso(),
        "blocks_pipeline": False,
        "smoke": bool(smoke),
        "note": json.dumps({"phase": phase, **note}, ensure_ascii=False),
    }
    R._write_json_atomic(path, payload)
    logger.info("[sentinel] wrote %s", path)
    return path


def _sampling_params(seed: int, *, plain: bool, smoke: bool):
    """#779-verbatim sampling params with PER-CELL render control (a natgen
    cell on a plain-listed rung must NOT get plain stop sequences)."""
    max_tokens = C.GEN_MAX_TOKENS if not smoke else min(C.GEN_MAX_TOKENS, R.SMOKE_GEN_MAX_TOKENS)
    stop = list(C.PLAIN_STOP_SEQUENCES) if plain else None
    if os.environ.get("EPM_ISSUE1902_GEN_ENGINE") == "hf":
        from types import SimpleNamespace

        return SimpleNamespace(
            n=1,
            temperature=C.GEN_TEMPERATURE,
            top_p=C.GEN_TOP_P,
            max_tokens=max_tokens,
            seed=seed,
            stop=stop,
        )
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=C.GEN_TEMPERATURE,
        top_p=C.GEN_TOP_P,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
    )


def _gen_cap(smoke: bool) -> int:
    return C.GEN_MAX_TOKENS if not smoke else min(C.GEN_MAX_TOKENS, R.SMOKE_GEN_MAX_TOKENS)


def effective_max_model_len(dims, *, smoke: bool) -> int:
    """Uniform max_model_len=8192 pin (plan §11); production FAILS LOUD on a
    rung whose max_pos cannot honor it (smoke tiny models shrink — a
    [smoke-downgrade], enumerated)."""
    if dims.max_position_embeddings >= C2.MAX_MODEL_LEN:
        return C2.MAX_MODEL_LEN
    if not smoke:
        raise RuntimeError(
            f"max_position_embeddings {dims.max_position_embeddings} < uniform "
            f"max_model_len {C2.MAX_MODEL_LEN} (plan §11 pin) — refusing"
        )
    print(
        f"[smoke-downgrade] max_model_len {C2.MAX_MODEL_LEN} -> "
        f"{dims.max_position_embeddings} (tiny model)",
        flush=True,
    )
    return dims.max_position_embeddings


def _config_dir(out_root: Path) -> Path:
    return out_root / "config"


def _stage_manifest_path(out_root: Path) -> Path:
    return out_root / "stage_manifest.json"


def load_stage_manifest(out_root: Path) -> dict[str, Any]:
    path = _stage_manifest_path(out_root)
    if not path.exists():
        raise FileNotFoundError(f"stage manifest missing: {path} — run `--phase stage` first")
    return R._read_json(path)


def _hf_download(
    filename: str,
    *,
    repo_id: str = C.HF_DATA_REPO,
    repo_type: str | None = "dataset",
    revision: str | None = None,
) -> Path:
    """Retried single-file Hub download (``hub.retry_transient`` routing —
    the live-HF-retry rule). Non-transient errors (EntryNotFound) propagate
    through the retry thunk unchanged."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    return Path(
        hub.retry_transient(
            lambda: hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision
            ),
            what=f"hf_hub_download {filename}",
        )
    )


def load_config_bundle(out_root: Path) -> dict[str, Any]:
    """Staged P0 config: exemplar bank, subsets, budget-recheck exclusions."""
    cdir = _config_dir(out_root)
    bundle = {
        "exemplars": R._read_json(cdir / "exemplars.json"),
        "subsets": R._read_json(cdir / "subsets.json"),
        "budget": R._read_json(cdir / "budget_recheck.json"),
    }
    bundle["excluded_ids"] = set(bundle["budget"]["excluded_ids"])
    return bundle


def load_rows(args: argparse.Namespace, out_root: Path) -> list[dict]:
    """Post-P0 corpus rows: the single-turn corpus minus budget-recheck
    violators (deterministic corpus order preserved)."""
    rows = R.load_corpus(Path(args.corpus_dir), C.CORPUS_SINGLE, smoke=args.smoke)
    excluded = load_config_bundle(out_root)["excluded_ids"]
    kept = [r for r in rows if r["id"] not in excluded]
    if excluded:
        logger.info("[rows] %d/%d rows kept (budget re-check exclusions)", len(kept), len(rows))
    return kept


def rows_for_scope(
    rows: list[dict],
    scope: str,
    cfg: dict[str, Any],
    isect_ids: list[str] | None = None,
) -> list[dict]:
    if scope == "full":
        sel = list(rows)
    elif scope == "intersection":
        assert isect_ids is not None, "intersection scope needs the P2.5 manifest ids"
        ids = set(isect_ids)
        sel = [r for r in rows if r["id"] in ids]
    else:
        ids = set(cfg["subsets"]["subsets"][scope])
        sel = [r for r in rows if r["id"] in ids]
    assert sel, f"empty row scope {scope!r} — mixed smoke/production config? ({len(rows)} rows)"
    return sel


def _rollout_path(out_root: Path, rung: str, cell: str) -> Path:
    return out_root / "gen" / rung / f"{cell}.jsonl"


def fetch_rollout(out_root: Path, rung: str, cell: str) -> list[dict]:
    """Rollout records, local-first with fail-loud HF fallback (fresh-pod
    resume: pass-2 consumes pass-1 rollouts; handles the text-shard layout)."""
    path = _rollout_path(out_root, rung, cell)
    if path.exists():
        return R._read_jsonl(path)
    from huggingface_hub.utils import EntryNotFoundError

    prefix = f"{C2.RAW_GEN_HF_PATH}/{rung}"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        got = _hf_download(f"{prefix}/{cell}.jsonl")
        path.write_bytes(got.read_bytes())
    except EntryNotFoundError:
        manifest = R._read_json(_hf_download(f"{prefix}/{cell}.manifest.json"))
        parts: list[bytes] = []
        for sh in manifest["shards"]:
            parts.append(_hf_download(f"{prefix}/{sh['name']}").read_bytes())
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_bytes(b"".join(parts))
        os.replace(tmp, path)
    C2.write_sha_sidecar(path)
    logger.info("[rollout] fetched %s/%s from HF (%d bytes)", rung, cell, path.stat().st_size)
    return R._read_jsonl(path)


def gen_diagnostics(recs: list[dict]) -> dict[str, Any]:
    """Per-cell degeneracy + channel-activity diagnostics from rollout records
    (plan §4 P2.5/M3): cap-hit, repetition rate, answer-length distribution,
    distinct-WORD ratio (documented operationalization of distinct-token
    ratio — no tokenizer pass), format-marker rate. Text never printed."""
    import numpy as np

    n = max(len(recs), 1)
    lens = np.array([r["n_tokens"] for r in recs] or [0], dtype=np.float64)
    distinct = []
    fmt = 0
    for r in recs:
        words = r["text"].split()
        distinct.append(len(set(words)) / max(len(words), 1))
        if C2._STRUCTURED_RE.search(r["text"]) or "```" in r["text"]:
            fmt += 1
    return {
        "n_rows": len(recs),
        "cap_hit_rate": sum(1 for r in recs if r["finish_reason"] == "length" or r["truncated"])
        / n,
        "truncated_rate": sum(1 for r in recs if r["truncated"]) / n,
        "repetition_rate": sum(1 for r in recs if r["repetition_flag"]) / n,
        "answer_len": {
            "mean": float(lens.mean()),
            "p10": float(np.percentile(lens, 10)),
            "p50": float(np.percentile(lens, 50)),
            "p90": float(np.percentile(lens, 90)),
        },
        "distinct_word_ratio_mean": float(np.mean(distinct)) if distinct else 0.0,
        "format_marker_rate": fmt / n,
    }


def answer_cloud_diagnostics(cell_dir: Path, layer: int) -> dict[str, Any]:
    """Effective rank + mean pairwise cosine of one layer's answer cloud
    (plan §4 P2.5), computed while the store is still local (pre-upload)."""
    import torch

    blob = torch.load(cell_dir / f"L{layer}.pt", weights_only=True)
    w = blob["w"].float()
    n, d = w.shape
    wc = w - w.mean(0, keepdim=True)
    s = torch.linalg.svdvals(wc)
    s2 = s.square()
    eff_rank = float(s2.sum().square() / s2.square().sum().clamp(min=1e-30))
    wn = torch.nn.functional.normalize(w, dim=1)
    m = wn.sum(0)
    mean_cos = float((m.dot(m) - n) / max(n * (n - 1), 1))
    return {
        "layer": layer,
        "n": n,
        "d": d,
        "effective_rank": eff_rank,
        "mean_pairwise_cos": mean_cos,
    }


def upload_cell_store(out_root: Path, subdir: str) -> dict[str, Any]:
    """Per-cell store upload AS WRITTEN -> verify -> delete-local (plan §4
    P3a/P3b; bounds in-flight store bytes). sha sidecars written FIRST so the
    consumed-shard hashes ride the upload (M2 hash-once)."""
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    store = R._store_root(out_root)
    cell_dir = store / subdir
    if not cell_dir.is_dir():
        raise FileNotFoundError(f"no store written under {cell_dir}")
    for f in sorted(cell_dir.rglob("*")):
        if f.is_file() and not f.name.endswith(".sha256"):
            C2.write_sha_sidecar(f)
    leaf_dirs = sorted({p.parent for p in cell_dir.rglob("*") if p.is_file()}, key=str)
    results: dict[str, Any] = {}
    for leaf in leaf_dirs:
        rel = leaf.relative_to(store).as_posix()
        res = upload_dir_sharded(
            leaf,
            C.HF_DATA_REPO,
            f"{C2.STORE_HF_PATH}/{rel}",
            repo_type="dataset",
            verify=True,
            delete_local=True,
        )
        results[rel] = {
            "uploaded": len(res.uploaded),
            "skipped_existing": len(res.skipped_existing),
            "rerouted": len(res.rerouted),
        }
        logger.info("[store-upload] %s: %s", rel, results[rel])
    return results


def _prompt_token_lens(tokenizer, prompts: list[str]) -> list[int]:
    return [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]]


# ── worker context (one loaded engine/model at a time per worker) ────────────


class _WorkerCtx:
    """Per-worker loaded-model cache: at most ONE vLLM engine OR HF capture
    model resident; switching rung/kind tears the previous one down (vLLM via
    the #1902 in-process reap; HF via del + empty_cache)."""

    def __init__(self, device: str, smoke: bool):
        self.device = device
        self.smoke = smoke
        self.rung: str | None = None
        self.kind: str | None = None
        self.engine = None
        self.model = None
        self._tokenizers: dict[str, Any] = {}

    def tokenizer(self, rung: str, pins: dict[str, str]):
        if rung not in self._tokenizers:
            self._tokenizers[rung] = R._tokenizer(C.MODEL_IDS[rung], C.resolve_revision(rung, pins))
        return self._tokenizers[rung]

    def release(self) -> None:
        if self.engine is not None:
            from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

            _reap_vllm_engine(self.engine)
            self.engine = None
        if self.model is not None:
            del self.model
            self.model = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cache release is best-effort on CPU hosts
            pass
        self.rung = self.kind = None

    def get_engine(self, rung: str, pins: dict[str, str]):
        if self.kind == "gen" and self.rung == rung and self.engine is not None:
            return self.engine
        self.release()
        dims = C.model_dims(C.MODEL_IDS[rung], C.resolve_revision(rung, pins))
        self.engine = R._vllm_engine(
            C.MODEL_IDS[rung],
            C.resolve_revision(rung, pins),
            effective_max_model_len(dims, smoke=self.smoke),
        )
        self.rung, self.kind = rung, "gen"
        return self.engine

    def get_model(self, rung: str, pins: dict[str, str]):
        if self.kind == "capture" and self.rung == rung and self.model is not None:
            return self.model
        self.release()
        self.model = R._load_hf_model(
            C.MODEL_IDS[rung], C.resolve_revision(rung, pins), self.device
        )
        self.rung, self.kind = rung, "capture"
        return self.model


# ── unit builders (fingerprinted at init; M2) ────────────────────────────────


def _hashes_bundle(out_root: Path) -> dict[str, Any]:
    sm = load_stage_manifest(out_root)
    return {
        "exemplars_sha": sm["exemplars_sha"],
        "subsets_sha": sm["subsets_sha"],
        "ladder_sha": sm["ladder_sha"],
        "corpus_sha": sm["corpus_sha"],
        "corpus_repo_revision": sm["corpus_repo_revision"],
    }


def _gen_fingerprint(
    cell: dict[str, Any], pins: dict[str, str], hashes: dict[str, Any], *, smoke: bool
) -> dict[str, Any]:
    return C2.build_fingerprint(
        "gen",
        code_sha=C2.code_sha(),
        rung=cell["rung"],
        revision=pins[cell["rung"]],
        render=cell["render"],
        k=cell["k"],
        order_id=cell["order_id"],
        set_id=cell["set_id"],
        seed=cell["seed"],
        sampling=C2.sampling_fingerprint(
            cell["seed"], plain=(cell["render"] == "plain"), max_tokens=_gen_cap(smoke)
        ),
        rows_scope=cell["rows_scope"],
        **hashes,
    )


def _capture1_base(
    cell: dict[str, Any], pins: dict[str, str], hashes: dict[str, Any], layers: list[int]
) -> dict[str, Any]:
    pooling = ["w", "u_last", "u_mean"] + (["q_mean"] if cell["want_q"] else [])
    return C2.build_fingerprint(
        "capture1",
        code_sha=C2.code_sha(),
        rung=cell["rung"],
        revision=pins[cell["rung"]],
        render=cell["render"],
        k=cell["k"],
        order_id=cell["order_id"],
        set_id=cell["set_id"],
        seed=cell.get("seed", C.GEN_SEED),  # reliability captures bind their gen seed
        sampling=None,  # gen regime rides transitively via rollout_sha
        rows_scope=cell["rows_scope"],
        rollout_sha=None,  # completed at execution from the write-time sidecar
        layers=list(layers),
        pooling_keys=pooling,
        **hashes,
    )


def _capture2_fingerprint(
    cell: dict[str, Any],
    pins: dict[str, str],
    hashes: dict[str, Any],
    layers: list[int],
    freeze_sha: str,
    intersection_sha: str,
) -> dict[str, Any]:
    pooling = ["w"] + (
        (["u_last", "u_mean"] + (["q_mean"] if cell["want_q"] else [])) if cell["store_ctx"] else []
    )
    return C2.build_fingerprint(
        "capture2",
        code_sha=C2.code_sha(),
        rung=cell["rung"],
        revision=pins[cell["rung"]],
        render=cell["render"],
        k=cell["k"],
        order_id=cell["order_id"],
        set_id=cell["set_id"],
        answer_source_rung=cell["answer_rung"],
        answer_source_revision=pins[cell["answer_rung"]],
        rollout_sha=None,  # completed at execution (consumed completion sha)
        layers=list(layers),
        pooling_keys=pooling,
        freeze_sha=freeze_sha,
        intersection_sha=intersection_sha,
        rows_scope=cell["rows_scope"],
        **hashes,
    )


def _resolve_layers(spec: str, rung: str, pins: dict[str, str], freeze: dict | None) -> list[int]:
    if spec == CAPTURE_LAYER_FULL:
        dims = C.model_dims(C.MODEL_IDS[rung], C.resolve_revision(rung, pins))
        return list(C.capture_layers(dims.num_layers))
    if spec == "band6":
        assert freeze is not None
        return [int(x) for x in freeze["band_b6"]]
    if spec == "lfa":
        assert freeze is not None
        return [int(freeze["layer_fa"])]
    if spec == "probe":
        dims = C.model_dims(C.MODEL_IDS[rung], C.resolve_revision(rung, pins))
        return R.probe_layers(dims.num_layers)
    raise ValueError(f"unknown layer spec {spec!r}")


# ── unit executors ───────────────────────────────────────────────────────────


def _exec_gen_unit(
    u: dict[str, Any],
    ctx: _WorkerCtx,
    rows: list[dict],
    cfg: dict[str, Any],
    pins: dict[str, str],
    out_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rung, cell = u["rung"], u["cell"]
    sel = rows_for_scope(rows, u["rows_scope"], cfg)
    turns = C2.exemplar_prefix_turns(cfg["exemplars"]["bank"], u["k"], u["order_id"], u["set_id"])
    tokenizer = ctx.tokenizer(rung, pins)
    if u["render"] == "plain":
        prompts = [C.render_plain_prompt(r["query"], turns) for r in sel]
    else:
        assert C.has_chat_template(tokenizer), f"native cell on template-less rung {rung}"
        assert turns is None, "native cells are 0-shot (plan §4 P2)"
        prompts = [C.render_chat_prompt(tokenizer, r["query"]) for r in sel]
    plens = _prompt_token_lens(tokenizer, prompts)
    llm = ctx.get_engine(rung, pins)
    sp = _sampling_params(u["seed"], plain=(u["render"] == "plain"), smoke=args.smoke)
    t0 = time.time()
    gens = R._generate_chunked(llm, prompts, sp)
    gen_wall = time.time() - t0
    recs = R._flag_records(sel, gens, u["seed"], gen_cap=_gen_cap(args.smoke))
    path = _rollout_path(out_root, rung, cell)
    R._write_jsonl_atomic(path, recs)
    rollout_sha = C2.write_sha_sidecar(path)
    # Persist-before-reduce (#779): text is on HF BEFORE any capture consumes it.
    R.upload_text_payload(path, f"{C2.RAW_GEN_HF_PATH}/{rung}")
    diag = gen_diagnostics(recs)
    import numpy as np

    info = {
        "rollout_sha": rollout_sha,
        "fingerprint_sha": C2.sha256_json({**u["fingerprint"], "rollout_sha": rollout_sha}),
        "gen_wall_s": round(gen_wall, 1),
        "rows_per_s": round(len(sel) / max(gen_wall, 1e-9), 3),
        "tok_per_s": round(sum(g["n_tokens"] for g in gens) / max(gen_wall, 1e-9), 1),
        "diagnostics": diag,
        # A1 per-(rung, k, render) over-window diagnostic + prompt-length dist.
        "over_window_frac": C2.over_window_fraction(plens),
        "prompt_len": {
            "p50": float(np.percentile(plens, 50)),
            "p95": float(np.percentile(plens, 95)),
            "max": int(max(plens)),
        },
    }
    return info


def _exec_capture_unit(
    u: dict[str, Any],
    ctx: _WorkerCtx,
    rows: list[dict],
    cfg: dict[str, Any],
    pins: dict[str, str],
    out_root: Path,
    args: argparse.Namespace,
    *,
    isect_ids: list[str] | None = None,
    upload: bool = True,
) -> dict[str, Any]:
    rung = u["rung"]
    sel = rows_for_scope(rows, u["rows_scope"], cfg, isect_ids=isect_ids)
    turns = C2.exemplar_prefix_turns(cfg["exemplars"]["bank"], u["k"], u["order_id"], u["set_id"])
    if turns is not None:
        sel = [{**r, "prefix_turns": turns} for r in sel]
    recs = fetch_rollout(out_root, u["answer_rung"], u["answer_cell"])
    rollout_sha = C2.read_sha_sidecar(_rollout_path(out_root, u["answer_rung"], u["answer_cell"]))
    answers = {r["id"]: r for r in recs}
    layers = u["layers_resolved"]
    tokenizer = ctx.tokenizer(rung, pins)
    model = ctx.get_model(rung, pins)
    stats = R.capture_cell(
        model,
        tokenizer,
        sel,
        answers,
        layers,
        out_root=out_root,
        ckpt=rung,
        src_label=u["answer_rung"],
        corpus=C.CORPUS_SINGLE,
        render=u["render"],
        device=args.device,
        store_subdir=u["subdir"],
        want_q=u["want_q"],
        store_ctx=u["store_ctx"],
        unit_tag=f" {u['unit']}",
    )
    mid_layer = layers[len(layers) // 2]
    cloud = answer_cloud_diagnostics(R._store_root(out_root) / u["subdir"], mid_layer)
    upload_res = upload_cell_store(out_root, u["subdir"]) if upload else {"skipped": True}
    return {
        "rollout_sha": rollout_sha,
        "fingerprint_sha": C2.sha256_json({**u["fingerprint"], "rollout_sha": rollout_sha}),
        "capture": stats,
        "answer_cloud": cloud,
        "upload": upload_res,
    }


# ── generic worker loop (M1 queue) ───────────────────────────────────────────


def worker_loop(
    args: argparse.Namespace,
    out_root: Path,
    phase: str,
    exec_unit,
) -> None:
    """Claim -> ensure snapshot -> execute -> mark done -> reap drained rungs.
    Fail fast: a unit exception marks the unit failed (unblocking the queue's
    dep sweep) and re-raises — the dispatcher sees the nonzero worker."""
    q = C2.UnitQueue(out_root, phase)
    pins = R.load_pins(out_root)
    wid = args.worker_id or f"w{os.getpid()}"
    ctx = _WorkerCtx(args.device, args.smoke)
    done_here = 0
    try:
        while True:
            claim = q.claim(wid, prefer_rung=ctx.rung, prefer_kind=ctx.kind)
            if claim is None:
                # Drain any orphaned reapables (a crashed sibling's drained
                # rung would otherwise hold a residency slot forever).
                for rung in q.take_reapable():
                    if ctx.rung == rung:
                        ctx.release()
                    C2.reap_snapshot(rung, pins, out_root)
                if q.all_terminal() or not q.any_running():
                    break
                time.sleep(15)
                continue
            name = claim["unit"]
            t0 = time.time()
            try:
                C2.ensure_snapshot(claim["rung"], pins, out_root)
                info = exec_unit(claim, ctx, pins)
                q.mark(name, "done", info)
                done_here += 1
                print(
                    f"[{phase}] unit {name} done (worker {wid}) elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            except BaseException as e:
                q.mark(name, "failed", {"error": f"{type(e).__name__}: {e}"})
                raise
            for rung in q.take_reapable():
                if ctx.rung == rung:
                    ctx.release()
                C2.reap_snapshot(rung, pins, out_root)
    finally:
        ctx.release()
    failed = q.failed_units()
    if failed:
        raise RuntimeError(f"[{phase}] {len(failed)} unit(s) failed/dep_failed: {sorted(failed)}")
    print(f"[{phase}] worker {wid} exiting clean ({done_here} units)", flush=True)


# ── P0: config build ─────────────────────────────────────────────────────────


def _stage_corpus_files(corpus_dir: Path) -> None:
    """Local-first corpus staging (corpus_single + clusters + manifest_stats)."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for name in (C.CORPUS_SINGLE_FILENAME, "clusters.json", "manifest_stats.json"):
        dest = corpus_dir / name
        if dest.exists():
            continue
        got = _hf_download(f"{C.CORPUS_HF_PATH}/{name}")
        dest.write_bytes(got.read_bytes())
        logger.info("[stage] fetched %s (%d bytes)", name, dest.stat().st_size)


def phase_config(args: argparse.Namespace, out_root: Path) -> None:
    """P0 (VM, CPU): budget re-check + exemplar bank (M3) + pinned subsets +
    ladder validation; fail-loud upload to ``issue2544_stage_map/config/``."""
    from transformers import AutoTokenizer

    corpus_dir = Path(args.corpus_dir)
    _stage_corpus_files(corpus_dir)
    rows = R.load_corpus(corpus_dir, C.CORPUS_SINGLE, smoke=args.smoke)
    cdir = _config_dir(out_root)
    cdir.mkdir(parents=True, exist_ok=True)

    # Ladder validation: the committed JSON is what the env points at.
    ladder_path = Path(os.environ["EPM_ISSUE1902_LADDER_JSON"])
    ladder = R._read_json(ladder_path)
    assert tuple(ladder["ckpts"]) == C.CKPTS, (tuple(ladder["ckpts"]), C.CKPTS)
    for c in C.CKPTS:
        assert C.MODEL_BRANCHES.get(c) == ladder["branches"].get(c), c
    logger.info("[config] ladder validated: %d rungs from %s", len(C.CKPTS), ladder_path)

    # P0 tokenizer load is pre-P1 UNPINNED by convention (C.default_revision_pins).
    tok_id = C.MODEL_IDS["main"]
    tokenizer = AutoTokenizer.from_pretrained(tok_id)

    # 1) Budget re-check at full consumed grain (Olmo-3 tokenizer).
    prompts = [C.render_plain_prompt(r["query"]) for r in rows]
    lens = _prompt_token_lens(tokenizer, prompts)
    violators = [
        {"id": r["id"], "n_tokens": n}
        for r, n in zip(rows, lens, strict=True)
        if n > C.MAX_FORMATTED_TOKENS
    ]
    budget = {
        "metadata": _metadata(),
        "tokenizer_id": tok_id,
        "n_rows": len(rows),
        "max_formatted_tokens": C.MAX_FORMATTED_TOKENS,
        "n_excluded": len(violators),
        "excluded_ids": [v["id"] for v in violators],
        "excluded_digest": violators,  # id + token count only — never row text
        "prompt_len_max": int(max(lens)),
    }
    R._write_json_atomic(cdir / "budget_recheck.json", budget)
    kept = [r for r in rows if r["id"] not in set(budget["excluded_ids"])]
    print(
        f"[config] budget re-check: {len(violators)} violators / {len(rows)} rows "
        f"(max prompt {budget['prompt_len_max']} tok)",
        flush=True,
    )

    # 2) Exemplar bank (M3: pool -> registered composition template).
    corpus_sha16s = {C2.sha16(r["query"]) for r in rows}
    quotas = C2.EXEMPLAR_POOL_QUOTAS if not args.smoke else SMOKE_EXEMPLAR_QUOTAS
    scan_cap = C2.EXEMPLAR_SCAN_CAP if not args.smoke else SMOKE_EXEMPLAR_SCAN_CAP
    if args.smoke:
        print(f"[smoke-downgrade] exemplar pool quotas {quotas}, scan cap {scan_cap}", flush=True)
    pool, pool_stats = C2.stream_exemplar_pool(
        tokenizer, corpus_sha16s, scan_cap=scan_cap, quotas=quotas
    )
    clusters = R._read_json(corpus_dir / "clusters.json")
    C2.assign_pool_clusters(pool, clusters["single"]["centroids"])
    bank = C2.select_exemplar_bank(pool)
    exemplars = {"metadata": _metadata(), "pool_stats": pool_stats, "bank": bank}
    R._write_json_atomic(cdir / "exemplars.json", exemplars)
    print(
        f"[config] exemplar bank: pool {pool_stats['counts']} "
        f"(scanned {pool_stats['scanned']}/{pool_stats['scan_cap']}); "
        f"sets composition {bank['composition']}",
        flush=True,
    )

    # 3) k16 render-fit assert (A16): longest realized k16 render + gen cap
    # must fit the uniform context pin. Checked on the 5 longest kept rows.
    k16_turns = C2.exemplar_prefix_turns(bank, 16, "O1", "S1")
    len_of = {r["id"]: n for r, n in zip(rows, lens, strict=True)}
    longest = sorted(kept, key=lambda r: len_of[r["id"]], reverse=True)[:5]
    k16_max = max(
        len(
            tokenizer.encode(C.render_plain_prompt(r["query"], k16_turns), add_special_tokens=False)
        )
        for r in longest
    )
    dims = C.model_dims(tok_id)
    fit_cap = effective_max_model_len(dims, smoke=args.smoke)
    assert k16_max + C.GEN_MAX_TOKENS <= fit_cap, (
        f"k16 render-fit VIOLATED: {k16_max} + {C.GEN_MAX_TOKENS} > {fit_cap}"
    )
    print(f"[config] k16 render-fit: {k16_max} + {C.GEN_MAX_TOKENS} <= {fit_cap} OK", flush=True)

    # 4) Pinned subsets (seed-42 class-stratified, post-exclusion rows).
    sizes = dict(C2.SUBSET_SIZES) if not args.smoke else dict(SMOKE_SUBSET_SIZES)
    if args.smoke:
        print(f"[smoke-downgrade] subset sizes {sizes}", flush=True)
    subsets = {label: C2.stratified_subset(kept, n, label) for label, n in sizes.items()}
    R._write_json_atomic(
        cdir / "subsets.json",
        {
            "metadata": _metadata(),
            "seed": C2.SUBSET_SEED,
            "sizes": {k: len(v) for k, v in subsets.items()},
            "subsets": subsets,
            "k16_render_max_tokens": k16_max,
        },
    )

    # Ladder copy travels with the config bundle (pods stage ONE prefix).
    (cdir / "issue2544_ladder.json").write_bytes(ladder_path.read_bytes())
    for name in ("budget_recheck.json", "exemplars.json", "subsets.json", "issue2544_ladder.json"):
        C2.write_sha_sidecar(cdir / name)

    # 5) Fail-loud upload of the config bundle.
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

    upload_dir_sharded(
        cdir,
        C.HF_DATA_REPO,
        C2.CONFIG_HF_PATH,
        repo_type="dataset",
        verify=True,
        delete_local=False,
    )
    phase_stage(args, out_root)  # config host records its own stage manifest
    write_sentinel(
        out_root,
        "config",
        {"n_rows": len(rows), "n_excluded": len(violators), "subset_sizes": sizes},
        smoke=args.smoke,
    )
    print("[phase=config] done", flush=True)


# ── stage: pod-side corpus + config staging ──────────────────────────────────


def phase_stage(args: argparse.Namespace, out_root: Path) -> None:
    """Stage corpus + P0 config bundle (local-first, HF fallback) and record
    the stage manifest — the ONE place corpus/config hashes are computed for
    every downstream fingerprint (M2 hash-once)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    corpus_dir = Path(args.corpus_dir)
    _stage_corpus_files(corpus_dir)
    cdir = _config_dir(out_root)
    cdir.mkdir(parents=True, exist_ok=True)
    for name in ("budget_recheck.json", "exemplars.json", "subsets.json", "issue2544_ladder.json"):
        dest = cdir / name
        if not dest.exists():
            got = _hf_download(f"{C2.CONFIG_HF_PATH}/{name}")
            dest.write_bytes(got.read_bytes())
            logger.info("[stage] fetched config/%s", name)

    corpus_file = corpus_dir / C.CORPUS_SINGLE_FILENAME
    repo_rev = hub.retry_transient(
        lambda: str(HfApi().dataset_info(C.HF_DATA_REPO).sha), what="dataset_info"
    )
    manifest = {
        "metadata": _metadata(),
        "corpus_sha": C2.sha256_file(corpus_file),
        "corpus_repo_revision": repo_rev,
        "exemplars_sha": C2.sha256_file(cdir / "exemplars.json"),
        "subsets_sha": C2.sha256_file(cdir / "subsets.json"),
        "budget_sha": C2.sha256_file(cdir / "budget_recheck.json"),
        "ladder_sha": C2.sha256_file(cdir / "issue2544_ladder.json"),
        "write_prefix": C2.HF_WRITE_PREFIX,
    }
    R._write_json_atomic(_stage_manifest_path(out_root), manifest)
    print(f"[phase=stage] manifest written: corpus {manifest['corpus_sha'][:12]}", flush=True)


# ── P1: pilot ────────────────────────────────────────────────────────────────


def _requested_rungs(args: argparse.Namespace) -> tuple[str, ...]:
    if not args.rungs:
        return C2.RUNGS
    toks = tuple(t for t in re.split(r"[,\s]+", args.rungs.strip()) if t)
    for t in toks:
        if t not in C2.RUNGS:
            raise SystemExit(f"unknown rung {t!r}; ladder: {C2.RUNGS}")
    return toks


def _tokenizer_json_sha(rung: str, pins: dict[str, str]) -> str:
    mid = C.MODEL_IDS[rung]
    if Path(mid).is_dir():
        p = Path(mid) / "tokenizer.json"
        return C2.sha256_file(p) if p.exists() else "absent-local"
    got = _hf_download("tokenizer.json", repo_id=mid, repo_type=None, revision=pins[rung])
    return C2.sha256_file(got)


def pilot_init(args: argparse.Namespace, out_root: Path) -> None:
    """P1 --init: BINDING pins, tokenizer-identity battery (A4), per-revision
    AutoConfig table incl. layer_types (A2), pilot unit queue."""
    pins = R.ensure_pins(out_root)
    rows = load_rows(args, out_root)
    rungs = _requested_rungs(args)

    # Tokenizer-identity battery: tokenizer.json hash + a 100-row encode probe
    # per pinned revision; ANY mismatch = designed halt (fairness control 8).
    probe_rows = rows[: min(100, len(rows))]
    battery: dict[str, dict[str, str]] = {}
    for rung in rungs:
        tok = R._tokenizer(C.MODEL_IDS[rung], C.resolve_revision(rung, pins))
        enc = tok([r["query"] for r in probe_rows], add_special_tokens=False)["input_ids"]
        battery[rung] = {
            "tokenizer_json_sha": _tokenizer_json_sha(rung, pins),
            "probe_ids_sha": C2.sha256_json(enc),
        }
    ref = battery[rungs[0]]
    mismatch = {r: b for r, b in battery.items() if b["probe_ids_sha"] != ref["probe_ids_sha"]}
    if mismatch:
        R.designed_halt(
            out_root,
            "tokenizer_identity",
            {"reference": rungs[0], "mismatching": sorted(mismatch), "battery": battery},
        )

    # Per-revision AutoConfig table (A2): ladder-wide shape invariance always;
    # Olmo-3 absolutes (max_pos >= 8192; full-attention layer set) production-only.
    from transformers import AutoConfig

    table: dict[str, dict[str, Any]] = {}
    for rung in rungs:
        c = AutoConfig.from_pretrained(C.MODEL_IDS[rung], revision=C.resolve_revision(rung, pins))
        lt = getattr(c, "layer_types", None)
        full_idx = sorted(i for i, t in enumerate(lt) if t == "full_attention") if lt else None
        table[rung] = {
            "num_layers": int(c.num_hidden_layers),
            "hidden": int(c.hidden_size),
            "max_pos": int(c.max_position_embeddings),
            "sliding_window": getattr(c, "sliding_window", None),
            "full_attention_layers": full_idx,
        }
    shapes = {(t["num_layers"], t["hidden"]) for t in table.values()}
    assert len(shapes) == 1, f"ladder shape drift: {table}"
    if not args.smoke:
        for rung, t in table.items():
            assert t["max_pos"] >= C2.MAX_MODEL_LEN, (rung, t)
            assert t["full_attention_layers"] == list(C2.OLMO3_FULL_ATTENTION_LAYERS), (rung, t)
    else:
        print("[smoke-downgrade] Olmo-3 absolute config asserts skipped (tiny model)", flush=True)

    hashes = _hashes_bundle(out_root)
    units: list[dict[str, Any]] = []
    for rung in rungs:
        for arm, k, oid, sid in (("gen0", 0, None, None), ("gen4", 4, "O1", "S1")):
            cell = {
                "cell": f"pilot_{arm}",
                "rung": rung,
                "render": "plain",
                "k": k,
                "order_id": oid,
                "set_id": sid,
                "seed": C.GEN_SEED,
                "rows_scope": "pilot",
            }
            units.append(
                {
                    "unit": f"pilotgen_{rung}_{arm}",
                    "rung": rung,
                    "kind": "gen",
                    "deps": [],
                    "fingerprint": _gen_fingerprint(cell, pins, hashes, smoke=args.smoke),
                    **cell,
                }
            )
        if rung in C2.PILOT_CAPTURE_RUNGS and rung in rungs:
            layers = _resolve_layers("probe", rung, pins, None)
            for arm, k, oid, sid in (("gen0", 0, None, None), ("gen4", 4, "O1", "S1")):
                cell = {
                    "cell": f"pilotcap_{arm}",
                    "rung": rung,
                    "render": "plain",
                    "k": k,
                    "order_id": oid,
                    "set_id": sid,
                    "answer_cell": f"pilot_{arm}",
                    "answer_rung": rung,
                    "rows_scope": "pilot",
                    "subdir": f"pilot/{rung}/{arm}",
                    "want_q": True,
                    "store_ctx": True,
                }
                units.append(
                    {
                        "unit": f"pilotcap_{rung}_{arm}",
                        "rung": rung,
                        "kind": "capture",
                        "deps": [f"pilotgen_{rung}_{arm}"],
                        "fingerprint": _capture1_base(cell, pins, hashes, layers),
                        "layers_resolved": layers,
                        **cell,
                    }
                )
    C2.UnitQueue(out_root, "pilot").init(units)
    R._write_json_atomic(
        out_root / "pilot_init.json",
        {
            "metadata": _metadata(),
            "rungs": list(rungs),
            "tokenizer_battery": battery,
            "config_table": table,
            "n_units": len(units),
        },
    )
    print(f"[phase=pilot] init: {len(units)} units, {len(rungs)} rungs", flush=True)


def _pilot_exec(args: argparse.Namespace, out_root: Path):
    cfg = load_config_bundle(out_root)
    rows = load_rows(args, out_root)

    def exec_unit(u: dict[str, Any], ctx: _WorkerCtx, pins: dict[str, str]) -> dict[str, Any]:
        if u["kind"] == "gen":
            return _exec_gen_unit(u, ctx, rows, cfg, pins, out_root, args)
        info = _exec_capture_unit(
            u, ctx, rows, cfg, pins, out_root, args, upload=False
        )  # pilot stores stay local (timed upload leg runs at finalize)
        import torch

        store = R._store_root(out_root)
        ctx_dir = store / u["subdir"] / "ctx"
        layer = u["layers_resolved"][len(u["layers_resolved"]) // 2]
        blob = torch.load(ctx_dir / f"L{layer}.pt", weights_only=True)
        if u["k"] == 0:
            # q_mean == u_mean at 0-shot: BIT-EXACT (identical masks) — the
            # plan P1/A14 tensor assert on real captured data.
            assert torch.equal(blob["q_mean"], blob["u_mean"]), "q_mean != u_mean at 0-shot"
            info["q_equals_u_asserted"] = True
        else:
            assert not torch.equal(blob["q_mean"], blob["u_mean"]), (
                "4-shot q_mean identical to u_mean — exemplar block not excluded?"
            )
            info["q_window_distinct_asserted"] = True
        if u["rung"] == "main" and u["cell"] == "pilotcap_gen0":
            # bf16 two-bar equivalence gate on Olmo-3 (finalize enforces).
            recs = fetch_rollout(out_root, u["rung"], u["answer_cell"])
            ok = [r for r in recs if not (r["truncated"] or r["repetition_flag"])]
            by_id = {r["id"]: r for r in ok}
            sel = [r for r in rows_for_scope(rows, "pilot", cfg) if r["id"] in by_id]
            entries = [
                R._capture_row_entry(
                    ctx.tokenizer(u["rung"], pins), r, by_id[r["id"]]["text"], render="plain"
                )
                for r in sel[: R.BF16_GATE_ROWS]
            ]
            model = ctx.get_model(u["rung"], pins)
            info["bf16_gate"] = R._bf16_equivalence_gate(
                model, entries, u["layers_resolved"], args.device
            )
        return info

    return exec_unit


def _wilson_lower(p_hat: float, n: int) -> float:
    if n == 0:
        return 0.0
    z2 = WILSON_Z**2
    denom = 1 + z2 / n
    center = p_hat + z2 / (2 * n)
    rad = WILSON_Z * ((p_hat * (1 - p_hat) + z2 / (4 * n)) / n) ** 0.5
    return max(0.0, (center - rad) / denom)


def _joint_intersection(
    flags: dict[tuple[str, str], set[str]], ids: list[str], rungs: tuple[str, ...]
) -> list[str]:
    """Row-level joint intersection: ids unflagged in BOTH arms at EVERY rung."""
    bad: set[str] = set()
    for rung in rungs:
        for arm in ("gen0", "gen4"):
            bad |= flags[(rung, arm)]
    return [i for i in ids if i not in bad]


def _gate_ladder(
    n_isect: int,
    n_rows: int,
    f_hat: float,
    lower_projected: float,
    tier_scan,
    *,
    gate: str,
    out_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    """The pre-registered Gate A / A' branch ladder (plan §4 P1).

    Returns the verdict dict on a PROCEED branch; designed-halts (rc=7)
    on (b) widen-reachable and (d). Under smoke the floors are informational
    (gate-calibration downgrade — logged, recorded, never a halt)."""
    projected = n_rows * f_hat
    verdict: dict[str, Any] = {
        "gate": gate,
        "n_isect": n_isect,
        "n_rows": n_rows,
        "f_hat": round(f_hat, 4),
        "projected": round(projected, 1),
        "projected_lower95": round(lower_projected, 1),
        "floor": C2.ISECT_FLOOR,
        "target": C2.ISECT_TARGET,
    }
    if lower_projected >= C2.ISECT_TARGET:
        verdict["branch"] = "a"
        return verdict
    if projected >= C2.ISECT_FLOOR:
        verdict["branch"] = "a_prime"
        verdict["caveat"] = "n < 2d — proceed with the registered caveat"
        return verdict
    needed = math_ceil_needed(f_hat)
    widenable = needed is not None and needed <= C2.WIDEN_BUILD_CAP
    verdict["widen_needed_build_n"] = needed
    if smoke:
        verdict["branch"] = "smoke-informational"
        print(f"[smoke-downgrade] {gate} floors informational: {verdict}", flush=True)
        return verdict
    if widenable:
        verdict["branch"] = "b"
        R.designed_halt(out_root, gate, {**verdict, "action": "widen corpus (VM rebuild) + re-run"})
    tier = tier_scan()
    if tier is not None:
        verdict["branch"] = "c"
        verdict["two_tier"] = tier
        return verdict
    verdict["branch"] = "d"
    R.designed_halt(out_root, gate, verdict)
    raise AssertionError("unreachable")  # designed_halt exits


def math_ceil_needed(f_hat: float) -> int | None:
    import math

    if f_hat <= 0:
        return None
    return int(math.ceil(C2.ISECT_FLOOR / f_hat) * 1.15)


def pilot_finalize(args: argparse.Namespace, out_root: Path) -> None:
    """P1 --finalize: Gate A on the pilot ROW-LEVEL joint intersection,
    cost re-projection, bf16 verdict, timed shard-upload leg, pilot report."""
    q = C2.UnitQueue(out_root, "pilot")
    if not q.all_terminal():
        raise RuntimeError("pilot finalize: units still pending/running")
    failed = q.failed_units()
    if failed:
        raise RuntimeError(f"pilot finalize: failed units {sorted(failed)}")
    snap = q.snapshot()
    pins = R.load_pins(out_root)
    cfg = load_config_bundle(out_root)
    rows = load_rows(args, out_root)
    rungs = tuple(dict.fromkeys(u["rung"] for u in snap["units"].values()))
    pilot_ids = [r["id"] for r in rows_for_scope(rows, "pilot", cfg)]

    flags: dict[tuple[str, str], set[str]] = {}
    for rung in rungs:
        for arm in ("gen0", "gen4"):
            recs = fetch_rollout(out_root, rung, f"pilot_{arm}")
            flags[(rung, arm)] = {r["id"] for r in recs if r["truncated"] or r["repetition_flag"]}
    isect = _joint_intersection(flags, pilot_ids, rungs)
    f_hat = len(isect) / max(len(pilot_ids), 1)
    n_corpus = len(rows)

    def _tier_scan() -> dict[str, Any] | None:
        # Concentration evidence (branch c): a suffix ladder r_min..R whose
        # joint intersection clears the floor at <= 2x widening, r_min <= r2.
        for i, r_min in enumerate(rungs[1:], start=1):
            if C2.RUNGS.index(r_min) > C2.RUNGS.index("r2"):
                break
            tier_rungs = rungs[i:]
            tier_isect = _joint_intersection(flags, pilot_ids, tier_rungs)
            tf = len(tier_isect) / max(len(pilot_ids), 1)
            if tf > 0 and (C2.ISECT_FLOOR / tf) <= 2 * n_corpus:
                return {
                    "r_min": r_min,
                    "tier_rungs": list(tier_rungs),
                    "floor_control_rungs": list(rungs[:i]),
                    "tier_f_hat": round(tf, 4),
                }
        return None

    gate_a = _gate_ladder(
        len(isect),
        n_corpus,
        f_hat,
        n_corpus * _wilson_lower(f_hat, len(pilot_ids)),
        _tier_scan,
        gate="gate_a",
        out_root=out_root,
        smoke=args.smoke,
    )

    # Cost re-projection: pilot-measured rates -> projected P2+P3 wall.
    infos = {n: u.get("info", {}) for n, u in snap["units"].items()}
    gen_rate = {
        u["rung"]: infos[n]["rows_per_s"]
        for n, u in snap["units"].items()
        if u["kind"] == "gen" and "rows_per_s" in infos[n]
    }
    cap_walls = [
        infos[n]["capture"]["per_row_wall_s"]
        for n, u in snap["units"].items()
        if u["kind"] == "capture" and "capture" in infos[n]
    ]
    sizes = cfg["subsets"]["sizes"]
    scope_n = {"full": n_corpus, **sizes}
    gen_cells = C2.gen_cell_roster(rungs)
    gen_rows_total = sum(scope_n[c["rows_scope"]] for c in gen_cells)
    gen_wall_s = sum(
        scope_n[c["rows_scope"]] / max(gen_rate.get(c["rung"], min(gen_rate.values())), 1e-6)
        for c in gen_cells
    )
    cap1_rows = sum(scope_n[c["rows_scope"]] for c in C2.pass1_capture_cells(rungs))
    cap2_rows = sum(
        scope_n.get(c["rows_scope"], len(isect))
        for c in C2.pass2_capture_cells(rungs, include_lfa0=True)
    )
    cap_wall_s = (cap1_rows + cap2_rows) * (sum(cap_walls) / max(len(cap_walls), 1))
    n_workers = 8
    projected_wall_h = (gen_wall_s + cap_wall_s) / 3600 / n_workers
    cost = {
        "gen_rows_total": gen_rows_total,
        "capture_rows_total": cap1_rows + cap2_rows,
        "gen_rate_rows_per_s": gen_rate,
        "capture_s_per_row_mean": sum(cap_walls) / max(len(cap_walls), 1),
        "projected_pass_wall_h_at_8w": round(projected_wall_h, 2),
        "booked_pass_wall_h": BOOKED_PASS_WALL_H,
    }
    if not args.smoke and projected_wall_h > C2.GATE_WALL_FACTOR * BOOKED_PASS_WALL_H:
        R.designed_halt(out_root, "capture_cost", cost)

    bf16 = next(
        (infos[n]["bf16_gate"] for n in infos if "bf16_gate" in infos[n]),
        None,
    )
    if bf16 is not None and not bf16["pass"] and not args.smoke:
        R.designed_halt(out_root, "bf16_equivalence", bf16)

    dims = C.model_dims(C.MODEL_IDS[rungs[-1]], C.resolve_revision(rungs[-1], pins))
    timing = R._timed_shard_upload(out_root, dims.hidden_size, args.smoke)

    init = R._read_json(out_root / "pilot_init.json")
    report = {
        "metadata": _metadata(),
        C.REVISION_PINS_KEY: pins,
        "gate_a": gate_a,
        "cost_projection": cost,
        "bf16_gate": bf16,
        "timing_shard": timing,
        "tokenizer_battery": init["tokenizer_battery"],
        "config_table": init["config_table"],
        "per_rung_survival": {
            rung: {
                arm: 1 - len(flags[(rung, arm)]) / max(len(pilot_ids), 1)
                for arm in ("gen0", "gen4")
            }
            for rung in rungs
        },
        "unit_infos": infos,
    }
    path = out_root / C.PILOT_REPORT_NAME
    R._write_json_atomic(path, report)
    R.upload_json_small(path, f"{C2.EVAL_MIRROR_HF_PATH}/pilot/{path.name}")
    write_sentinel(
        out_root,
        "pilot",
        {"gate_a": gate_a["branch"], "projected": gate_a["projected"]},
        smoke=args.smoke,
    )
    print(f"[phase=pilot] done: gate A branch {gate_a['branch']}", flush=True)


def phase_pilot(args: argparse.Namespace, out_root: Path) -> None:
    if args.init:
        pilot_init(args, out_root)
        return
    if args.finalize:
        pilot_finalize(args, out_root)
        return
    exec_unit = _pilot_exec(args, out_root)
    worker_loop(args, out_root, "pilot", exec_unit)


# ── P2+P3a: pass 1 ───────────────────────────────────────────────────────────


def pass1_init(args: argparse.Namespace, out_root: Path) -> None:
    report = out_root / C.PILOT_REPORT_NAME
    if not report.exists():
        raise RuntimeError(f"pilot report missing: {report} — run --phase pilot first")
    pins = R.load_pins(out_root)
    hashes = _hashes_bundle(out_root)
    rungs = _requested_rungs(args)
    units: list[dict[str, Any]] = []
    for cell in C2.gen_cell_roster(rungs):
        units.append(
            {
                "unit": f"gen_{cell['rung']}_{cell['cell']}",
                "rung": cell["rung"],
                "kind": "gen",
                "deps": [],
                "fingerprint": _gen_fingerprint(cell, pins, hashes, smoke=args.smoke),
                **cell,
            }
        )
    for cell in C2.pass1_capture_cells(rungs):
        layers = _resolve_layers(cell.pop("layers"), cell["rung"], pins, None)
        units.append(
            {
                "unit": f"cap_{cell['rung']}_{cell['cell']}",
                "rung": cell["rung"],
                "kind": "capture",
                "deps": [f"gen_{cell['answer_rung']}_{cell['answer_cell']}"],
                "fingerprint": _capture1_base(cell, pins, hashes, layers),
                "layers_resolved": layers,
                **cell,
            }
        )
    R.headroom_gate(out_root, "capture", len(units), R.CAPTURE_PER_CELL_GB)
    C2.UnitQueue(out_root, "pass1").init(units)
    print(f"[phase=pass1] init: {len(units)} units across {len(rungs)} rungs", flush=True)


def pass1_finalize(args: argparse.Namespace, out_root: Path) -> None:
    """P2.5: degeneracy + channel-activity diagnostics per (rung, cell), the
    realized joint shared intersection, fold table + Gate A'."""
    q = C2.UnitQueue(out_root, "pass1")
    if not q.all_terminal():
        raise RuntimeError("pass1 finalize: units still pending/running")
    failed = q.failed_units()
    if failed:
        raise RuntimeError(f"pass1 finalize: failed units {sorted(failed)}")
    snap = q.snapshot()
    cfg = load_config_bundle(out_root)
    rows = load_rows(args, out_root)
    rungs = tuple(dict.fromkeys(u["rung"] for u in snap["units"].values()))
    all_ids = [r["id"] for r in rows]

    diagnostics: dict[str, Any] = {}
    flags: dict[tuple[str, str], set[str]] = {}
    for rung in rungs:
        diagnostics[rung] = {}
        for arm in ("gen0", "gen4"):
            recs = fetch_rollout(out_root, rung, arm)
            flags[(rung, arm)] = {r["id"] for r in recs if r["truncated"] or r["repetition_flag"]}
            diagnostics[rung][arm] = gen_diagnostics(recs)
        d0, d4 = diagnostics[rung]["gen0"], diagnostics[rung]["gen4"]
        diagnostics[rung]["channel_activity_shift_4v0"] = {
            "answer_len_mean": d4["answer_len"]["mean"] - d0["answer_len"]["mean"],
            "format_marker_rate": d4["format_marker_rate"] - d0["format_marker_rate"],
            "cap_hit_rate": d4["cap_hit_rate"] - d0["cap_hit_rate"],
            "distinct_word_ratio": d4["distinct_word_ratio_mean"] - d0["distinct_word_ratio_mean"],
        }

    isect = _joint_intersection(flags, all_ids, rungs)
    f_hat = len(isect) / max(len(all_ids), 1)

    def _tier_scan() -> dict[str, Any] | None:
        for i, r_min in enumerate(rungs[1:], start=1):
            if C2.RUNGS.index(r_min) > C2.RUNGS.index("r2"):
                break
            tier = _joint_intersection(flags, all_ids, rungs[i:])
            if len(tier) >= C2.ISECT_FLOOR:
                return {
                    "r_min": r_min,
                    "tier_rungs": list(rungs[i:]),
                    "floor_control_rungs": list(rungs[:i]),
                    "n_tier": len(tier),
                    "tier_ids": tier,
                }
        return None

    gate = _gate_ladder(
        len(isect),
        len(all_ids),
        f_hat,
        float(len(isect)),  # realized — no projection band at P2.5
        _tier_scan,
        gate="gate_a_prime",
        out_root=out_root,
        smoke=args.smoke,
    )
    headline_ids = gate.get("two_tier", {}).get("tier_ids", isect)

    # Fold table (6 cluster-grouped folds; re-balance on a min-n_tr violation).
    group_of = {r["id"]: r["group"] for r in rows}
    groups = [group_of[i] for i in headline_ids]
    best = None
    for trial in range(20):
        assign = R.assign_fold_groups(groups, C.N_FOLDS, seed=C.FOLD_SEED + trial)
        fold_sizes = [0] * C.N_FOLDS
        for g in groups:
            fold_sizes[assign[g]] += 1
        min_ntr = len(headline_ids) - max(fold_sizes)
        if best is None or min_ntr > best["min_ntr"]:
            best = {
                "seed": C.FOLD_SEED + trial,
                "fold_sizes": fold_sizes,
                "min_ntr": min_ntr,
                "assign": assign,
            }
        if min_ntr >= C2.FOLD_MIN_NTR:
            break
    assert best is not None
    if best["min_ntr"] < C2.FOLD_MIN_NTR:
        if args.smoke:
            print(f"[smoke-downgrade] fold min n_tr {best['min_ntr']} informational", flush=True)
        else:
            R.designed_halt(
                out_root,
                "gate_a_prime",
                {**gate, "fold_min_ntr": best["min_ntr"], "floor": C2.FOLD_MIN_NTR},
            )

    input_fps = {
        n: u.get("info", {}).get("fingerprint_sha")
        for n, u in snap["units"].items()
        if u["kind"] == "gen"
    }
    over_window = {
        n: {
            "over_window_frac": u["info"].get("over_window_frac"),
            "prompt_len": u["info"].get("prompt_len"),
            "rung": u["rung"],
            "k": u.get("k"),
            "render": u.get("render"),
        }
        for n, u in snap["units"].items()
        if u["kind"] == "gen" and "info" in u and "over_window_frac" in u.get("info", {})
    }
    manifest = {
        "metadata": _metadata(),
        "gate_a_prime": {k: v for k, v in gate.items() if k != "two_tier"},
        "two_tier": gate.get("two_tier", None)
        and {k: v for k, v in gate["two_tier"].items() if k != "tier_ids"},
        "n_isect_all_rungs": len(isect),
        "headline_ids": headline_ids,
        "fold_table": {k: best[k] for k in ("seed", "fold_sizes", "min_ntr")},
        "fold_assign": best["assign"],
        "per_rung_survival": {
            rung: {
                arm: 1 - len(flags[(rung, arm)]) / max(len(all_ids), 1) for arm in ("gen0", "gen4")
            }
            for rung in rungs
        },
        "input_fingerprints": input_fps,
        "over_window_diagnostics": over_window,
    }
    gen_dir = out_root / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(gen_dir / "intersection_manifest.json", manifest)
    C2.write_sha_sidecar(gen_dir / "intersection_manifest.json")
    R._write_json_atomic(gen_dir / "diagnostics.json", {"metadata": _metadata(), **diagnostics})
    R.upload_json_small(
        gen_dir / "intersection_manifest.json",
        f"{C2.EVAL_MIRROR_HF_PATH}/gen/intersection_manifest.json",
    )
    R.upload_json_small(
        gen_dir / "diagnostics.json", f"{C2.EVAL_MIRROR_HF_PATH}/gen/diagnostics.json"
    )
    write_sentinel(
        out_root,
        "pass1",
        {"gate_a_prime": gate["branch"], "n_isect": len(isect)},
        smoke=args.smoke,
    )
    print(
        f"[phase=pass1] done: gate A' branch {gate['branch']} "
        f"(n_isect={len(isect)}, min_ntr={best['min_ntr']})",
        flush=True,
    )


def phase_pass1(args: argparse.Namespace, out_root: Path) -> None:
    if args.init:
        pass1_init(args, out_root)
        return
    if args.finalize:
        pass1_finalize(args, out_root)
        return
    cfg = load_config_bundle(out_root)
    rows = load_rows(args, out_root)

    def exec_unit(u: dict[str, Any], ctx: _WorkerCtx, pins: dict[str, str]) -> dict[str, Any]:
        if u["kind"] == "gen":
            return _exec_gen_unit(u, ctx, rows, cfg, pins, out_root, args)
        return _exec_capture_unit(u, ctx, rows, cfg, pins, out_root, args)

    worker_loop(args, out_root, "pass1", exec_unit)


# ── P3b: pass 2 ──────────────────────────────────────────────────────────────


def _freeze_record(out_root: Path) -> tuple[dict[str, Any], str]:
    """P4a layer-freeze record (Unit B's output): layer*, band B5, l_FA, band
    B6. Local-first; HF eval-mirror fallback; fail-loud with both paths."""
    path = out_root / "fits" / "layer_freeze.json"
    if not path.exists():
        try:
            got = _hf_download(f"{C2.EVAL_MIRROR_HF_PATH}/fits/layer_freeze.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(got.read_bytes())
        except Exception as e:
            raise RuntimeError(
                f"layer-freeze record missing: {path} (and not on HF at "
                f"{C2.EVAL_MIRROR_HF_PATH}/fits/layer_freeze.json) — run P4a "
                f"(issue2544_fits.py) first: {type(e).__name__}: {e}"
            ) from e
    freeze = R._read_json(path)
    for key in ("layer_star", "band_b5", "layer_fa", "band_b6"):
        if key not in freeze:
            raise RuntimeError(f"layer_freeze.json missing key {key!r}: {sorted(freeze)}")
    expected_fa = C2.nearest_full_attention_layer(int(freeze["layer_star"]))
    assert int(freeze["layer_fa"]) == expected_fa or Path(C.MODEL_IDS[C2.RUNGS[0]]).is_dir(), (
        freeze["layer_fa"],
        expected_fa,
    )
    assert sorted(set(freeze["band_b5"]) | {freeze["layer_fa"]}) == sorted(freeze["band_b6"]), (
        freeze["band_b5"],
        freeze["layer_fa"],
        freeze["band_b6"],
    )
    sha = C2.sha256_file(path)
    return freeze, sha


def _intersection(out_root: Path) -> tuple[dict[str, Any], str]:
    path = out_root / "gen" / "intersection_manifest.json"
    if not path.exists():
        raise RuntimeError(f"intersection manifest missing: {path} — run pass1 --finalize first")
    return R._read_json(path), C2.sha256_file(path)


def pass2_init(args: argparse.Namespace, out_root: Path) -> None:
    pins = R.load_pins(out_root)
    hashes = _hashes_bundle(out_root)
    rungs = _requested_rungs(args)
    freeze, freeze_sha = _freeze_record(out_root)
    manifest, isect_sha = _intersection(out_root)
    dims = C.model_dims(C.MODEL_IDS[rungs[0]], C.resolve_revision(rungs[0], pins))
    pass1_layers = set(C.capture_layers(dims.num_layers))
    include_lfa0 = int(freeze["layer_fa"]) not in pass1_layers
    if not include_lfa0:
        print(
            f"[phase=pass2] l_FA={freeze['layer_fa']} already in pass-1 set — lfa0 skipped",
            flush=True,
        )
    units: list[dict[str, Any]] = []
    for cell in C2.pass2_capture_cells(rungs, include_lfa0=include_lfa0):
        layers = _resolve_layers(cell.pop("layers"), cell["rung"], pins, freeze)
        units.append(
            {
                "unit": f"cap2_{cell['rung']}_{cell['cell']}",
                "rung": cell["rung"],
                "kind": "capture",
                "deps": [],  # answers are pass-1 rollout FILES (fetched fail-loud)
                "fingerprint": _capture2_fingerprint(
                    cell, pins, hashes, layers, freeze_sha, isect_sha
                ),
                "layers_resolved": layers,
                **cell,
            }
        )
    R.headroom_gate(out_root, "capture", len(units), R.CAPTURE_PER_CELL_GB)
    C2.UnitQueue(out_root, "pass2").init(units)
    print(f"[phase=pass2] init: {len(units)} units (band B6 = {freeze['band_b6']})", flush=True)


def pass2_finalize(args: argparse.Namespace, out_root: Path) -> None:
    q = C2.UnitQueue(out_root, "pass2")
    if not q.all_terminal():
        raise RuntimeError("pass2 finalize: units still pending/running")
    failed = q.failed_units()
    if failed:
        raise RuntimeError(f"pass2 finalize: failed units {sorted(failed)}")
    snap = q.snapshot()
    summary = {
        "metadata": _metadata(),
        "n_units": len(snap["units"]),
        "units": {n: u.get("info", {}).get("capture", {}) for n, u in snap["units"].items()},
    }
    path = out_root / "pass2_summary.json"
    R._write_json_atomic(path, summary)
    R.upload_json_small(path, f"{C2.EVAL_MIRROR_HF_PATH}/capture/pass2_summary.json")
    write_sentinel(out_root, "pass2", {"n_units": len(snap["units"])}, smoke=args.smoke)
    print(f"[phase=pass2] done: {len(snap['units'])} units", flush=True)


def phase_pass2(args: argparse.Namespace, out_root: Path) -> None:
    if args.init:
        pass2_init(args, out_root)
        return
    if args.finalize:
        pass2_finalize(args, out_root)
        return
    cfg = load_config_bundle(out_root)
    rows = load_rows(args, out_root)
    manifest, _ = _intersection(out_root)
    isect_ids = manifest["headline_ids"]

    def exec_unit(u: dict[str, Any], ctx: _WorkerCtx, pins: dict[str, str]) -> dict[str, Any]:
        return _exec_capture_unit(u, ctx, rows, cfg, pins, out_root, args, isect_ids=isect_ids)

    worker_loop(args, out_root, "pass2", exec_unit)


# ── fits registration point (Unit B) ─────────────────────────────────────────


def phase_fits(args: argparse.Namespace, out_root: Path) -> None:
    fits_path = _SCRIPTS_DIR / "issue2544_fits.py"
    if not fits_path.exists():
        print(
            "[phase=fits] scripts/issue2544_fits.py not present yet (Unit B "
            "deliverable) — this entrypoint is the registration point.",
            flush=True,
        )
        raise SystemExit(2)
    from issue2544_fits import run_fits

    run_fits(args, out_root)


# ── import check / main ──────────────────────────────────────────────────────


def _import_check() -> None:
    """Execute every deferred import + the argparse-attribute/bind pass."""
    import datasets  # noqa: F401
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    import vllm  # noqa: F401
    from datasets import load_dataset  # noqa: F401
    from huggingface_hub import (  # noqa: F401
        HfApi,
        hf_hub_download,
        scan_cache_dir,
        snapshot_download,
    )
    from huggingface_hub.utils import EntryNotFoundError  # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401
    from transformers import AutoConfig, AutoTokenizer  # noqa: F401
    from vllm import SamplingParams  # noqa: F401

    from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: F401
    from explore_persona_space.analysis.representation_shift import (  # noqa: F401
        _reap_vllm_engine,
        compute_prompt_spans,
    )
    from explore_persona_space.eval.generation import create_vllm_engine  # noqa: F401
    from explore_persona_space.orchestrate import hub  # noqa: F401
    from explore_persona_space.orchestrate.preflight import (  # noqa: F401
        assert_out_root_headroom,
    )
    from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded  # noqa: F401

    if (_SCRIPTS_DIR / "issue2544_fits.py").exists():
        import issue2544_fits  # noqa: F401
    else:
        print("[import-check] issue2544_fits.py not present yet (Unit B) — skipped", flush=True)
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__, str(Path(C2.__file__).resolve()))
    print("[import-check] issue2544_run OK", flush=True)


def _default_out_root() -> Path:
    env = os.environ.get("EPM_ISSUE2544_OUT_ROOT")
    if env:
        return Path(env)
    ws = Path("/workspace")
    if ws.is_dir():
        return ws / "issue2544"
    return PROJECT_ROOT / "data" / "issue_2544" / "out"


PHASES = ("config", "stage", "pilot", "pass1", "pass2", "fits")


def main() -> None:
    ap = argparse.ArgumentParser(description="#2544 Olmo-3 stage-map gen+capture driver")
    ap.add_argument("--phase", choices=list(PHASES))
    ap.add_argument("--init", action="store_true", help="phase init leg (queue/pins)")
    ap.add_argument("--worker", action="store_true", help="phase worker leg (unit queue)")
    ap.add_argument("--finalize", action="store_true", help="phase finalize leg")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-root", default=str(_default_out_root()))
    ap.add_argument("--corpus-dir", default=None, help="default: <out-root>/corpus")
    ap.add_argument("--rungs", default=None, help="rung subset, e.g. 'r0 main R' (default: all)")
    ap.add_argument("--worker-id", default=None)
    ap.add_argument("--gpu-id", type=int, default=None, help="recorded; CVD pinned by dispatcher")
    ap.add_argument("--device", default=None, help="torch device (default cuda:0 if available)")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    args = ap.parse_args()

    if args.list_phases:
        print(" ".join(sorted(PHASES)))
        sys.exit(0)
    if args.import_check:
        _import_check()
        sys.exit(0)
    if not args.phase:
        ap.error("--phase is required (or --list-phases / --import-check)")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.corpus_dir is None:
        args.corpus_dir = str(out_root / "corpus")
    if args.device is None:
        # #1902 parity: the dispatcher composes no --device, so the CPU-host
        # smoke (tiny-real model) resolves cpu here; cuda hosts keep cuda:0.
        try:
            import torch

            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 — CPU structural checks without torch cuda
            args.device = "cpu"
    if args.smoke and C2.HF_WRITE_PREFIX == "issue2544_stage_map":
        raise SystemExit(
            "--smoke refuses the PRODUCTION HF write prefix — export "
            "EPM_ISSUE1902_HF_WRITE_PREFIX=issue2544_stage_map/_smoke first "
            "(smoke outputs never overwrite committed artifacts)"
        )
    for rung in _requested_rungs(args):
        _ = C.MODEL_IDS[rung]  # fail loud on an unmapped rung before any work

    print(f"[phase={args.phase}] out_root={out_root} smoke={args.smoke}", flush=True)
    dispatch = {
        "config": phase_config,
        "stage": phase_stage,
        "pilot": phase_pilot,
        "pass1": phase_pass1,
        "pass2": phase_pass2,
        "fits": phase_fits,
    }
    dispatch[args.phase](args, out_root)
    # Explicit clean exit (PyGILState atexit rule — never fall off main).
    sys.exit(0)


if __name__ == "__main__":
    main()
