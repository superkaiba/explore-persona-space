"""Issue #2221 P1 — real-data corpus staging (prompts, panel rollouts, found responses).

Phases (``--phase``; registry ``PHASES``):

- ``prompts``     : extract the paper's REAL user prompts per EM-like family from
                    the persona_vectors dataset release (``external/persona_vectors``).
- ``found``       : stream LMSYS-Chat-1M + WildChat-1M, keep real single-exchange
                    (user, assistant) pairs (tier-1 deployment data), near-dup
                    screened — the chat-trait candidate pool.
- ``cvefixes``    : stream CVEfixes; keep CVSS + vulnerable/fixed code fields
                    (fail loud when the schema probe resolves neither — plan A12).
- ``panel_prompts``: seeded draw of the LMSYS real-prompt eval panel (P5/P6 surface).
- ``rollouts``    : vLLM sampling of the non-Qwen panel (~6 rollouts/prompt,
                    temp 1.0, ``max_new_tokens`` 1024, seed 0) over the EM-family
                    prompts; per-(family, model) cap-hit fraction reported with the
                    >2% re-gen trigger.
- ``rollouts_regen``: the trigger's ACTION arm (v14): for every cell whose meta
                    carries ``regen_trigger: true``, re-generate ONLY the
                    truncated rows (persisted ``finish_reason == "length"``) at
                    >=2x the cap and splice them in place (stable row identity;
                    one pass, residual reported — never cap iteration).
- ``upload``      : persist ALL staged text to the HF data repo under
                    ``issue2221_realtwin/raw_completions/{rollouts,found}/...``.

Content hygiene: raw LMSYS/WildChat/rollout text is NEVER printed or logged —
digests only (row counts, sha256, per-filter reject counters).

GPU phases (``rollouts``) run pod-side; multi-GB pulls stage under ``--out-root``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import os  # noqa: E402

# vLLM v1 EngineCore dies silently under fork() when the parent touched
# tokenizers before LLM() (gotchas.md #628) — set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue778_lib as lib  # noqa: E402

from explore_persona_space.experiments.issue_1739.corpus_staging import (  # noqa: E402
    _fingerprint,
    _hf_stream,
    _stream_stage,
    minhash_signatures,
    usable_text,
)
from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue_2221.loaders import (  # noqa: E402
    atomic_write_text,
    read_jsonl,
    resume_ok,
    self_near_dup_mask,
    write_fingerprint,
    write_jsonl,
)

logger = logging.getLogger("issue2221.stage")

_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(model_id: str):
    """Module-cached tokenizer load (never from_pretrained inside a loop)."""
    if model_id not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _TOKENIZER_CACHE[model_id]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "_").lower()


# ── per-model vLLM attention-kernel env pins (P1 attempt-2 crash fix) ─────────
#
# gemma-2 uses tanh LOGIT SOFTCAPPING in attention; the FA3 kernel vLLM 0.11.0
# selects by default on H100 (cc 9.0 — fa_utils.get_flash_attn_version step 1)
# is built WITHOUT softcap support and dies at engine init during cudagraph
# warmup: RuntimeError "This flash attention build does not support tanh
# softcapping" (torch.ops._vllm_fa3_C.fwd). FA2 supports softcapping
# (_vllm_fa2_C.varlen_fwd takes softcap=) and the env override wins version
# selection (fa_utils step 2: VLLM_FLASH_ATTN_VERSION in {2,3}). FLASHINFER
# was rejected: flashinfer is NOT installed on pod-2221 (site-packages probed
# 2026-08-11). vLLM env vars are read LAZILY at attribute access
# (vllm/envs.py __getattr__), so setting the env after `import vllm` but
# BEFORE build_vllm_engine() is effective; the spawned EngineCore worker
# inherits the parent's os.environ.
_GEMMA2_ATTN_ENV = {"VLLM_FLASH_ATTN_VERSION": "2"}


def _attn_env_overrides(model_id: str) -> dict[str, str]:
    """Per-model vLLM engine env pins; empty for every non-gemma-2 model."""
    if "gemma-2" in model_id.lower():
        return dict(_GEMMA2_ATTN_ENV)
    return {}


def _apply_attn_env(model_id: str) -> list[str]:
    """Set the model's attention env pins (setdefault semantics).

    Returns the keys THIS call set, so ``_restore_attn_env`` never pops a
    launcher-provided value. Logs the effective pin — the fix-engaged signal
    for the gemma-2 relaunch.
    """
    applied: list[str] = []
    for k, v in _attn_env_overrides(model_id).items():
        if os.environ.get(k) is None:
            os.environ[k] = v
            applied.append(k)
        lib.log_phase(
            "p1_rollouts",
            f"{_model_slug(model_id)}: attention-kernel pin {k}={os.environ[k]}"
            + ("" if k in applied else " (launcher-provided, kept)"),
        )
    return applied


def _restore_attn_env(applied: list[str]) -> None:
    """Pop only the keys _apply_attn_env set (non-gemma engines stay byte-identical)."""
    for k in applied:
        os.environ.pop(k, None)


# ── prompts (paper dataset user turns) ────────────────────────────────────────


def ensure_paper_repo(external_root: Path) -> Path:
    """Clone the persona_vectors release if absent; extract dataset.zip if needed.

    Upstream (safety-research/persona_vectors) ships the data ONLY as
    ``dataset.zip`` — no ``dataset/`` in the tree — so a fresh clone can never
    satisfy a bare ``dataset/`` assert (v5 pod crash, defect 1). Idempotent:
    an existing ``dataset/`` early-returns; fail loud when neither the dir nor
    the zip exists post-clone.
    """
    import subprocess
    import zipfile

    ds = external_root / "dataset"
    if ds.is_dir():
        return ds
    if not external_root.is_dir() or not any(external_root.iterdir()):
        external_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/safety-research/persona_vectors",
                str(external_root),
            ],
            check=True,
            env={**os.environ},
        )
    if not ds.is_dir():
        zip_path = external_root / "dataset.zip"
        if not zip_path.is_file():
            raise RuntimeError(
                f"persona_vectors clone has neither dataset/ nor dataset.zip at {external_root}"
            )
        logger.info("[p1_prompts] extracting %s (upstream ships the data zipped)", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(external_root)
    if not ds.is_dir():
        raise RuntimeError(f"persona_vectors dataset.zip did not produce dataset/ at {ds}")
    return ds


def phase_prompts(args) -> None:
    """Extract per-EM-family REAL user prompts from the paper dataset release."""
    ds_root = ensure_paper_repo(Path(args.external_root))
    out = Path(args.out_root) / "prompts"
    for family in C.EM_FAMILIES:
        src = ds_root / family / "normal.jsonl"
        if not src.is_file():
            raise FileNotFoundError(f"paper dataset file missing: {src}")
        rows = read_jsonl(src)
        prompts: list[dict] = []
        seen: set[str] = set()
        for r in rows:
            msgs = r.get("messages", [])
            if not msgs or msgs[0].get("role") != "user":
                raise ValueError(f"{family}: unexpected paper row shape (first role != user)")
            content = msgs[0].get("content")
            # usable_text returns the REJECT REASON, or None when usable (#1739
            # helper contract) — keep on None, stage the actual content.
            reason = usable_text(content)
            if reason is not None:
                continue
            text = content.strip()
            h = _sha(text)
            if h in seen:
                continue
            seen.add(h)
            prompts.append({"idx": len(prompts), "sha": h, "prompt": text})
            if len(prompts) >= args.prompts_cap:
                break
        if not prompts:
            raise RuntimeError(f"{family}: 0 prompts extracted")
        write_jsonl(out / f"{family}.jsonl", prompts)
        lib.log_phase("p1_prompts", f"{family}: {len(prompts)} prompts staged")


# ── found responses (LMSYS + WildChat) ────────────────────────────────────────


def _first_exchange(conv: list[dict]) -> tuple[str, str] | None:
    """Reduce a conversation to its FIRST (user, assistant) exchange."""
    for i in range(len(conv) - 1):
        a, b = conv[i], conv[i + 1]
        if a.get("role") == "user" and b.get("role") == "assistant":
            u, v = a.get("content"), b.get("content")
            # usable_text returns the REJECT REASON, or None when usable —
            # keep the exchange only when BOTH members are usable.
            if usable_text(u) is None and usable_text(v) is None:
                return u.strip(), v.strip()
            return None
    return None


def _keep_lmsys(row: dict) -> tuple[dict | None, str | None]:
    # LMSYS stores FULL language names ('English'), never ISO codes (#1092).
    if row.get("language") != "English":
        return None, "language"
    mod = row.get("openai_moderation") or []
    if any(isinstance(m, dict) and m.get("flagged") for m in mod):
        return None, "moderation_flagged"
    ex = _first_exchange(row.get("conversation") or [])
    if ex is None:
        return None, "no_single_exchange"
    prompt, response = ex
    return {
        "id": f"lmsys-{_sha(prompt + chr(30) + response)}",
        "corpus": "lmsys",
        "prompt": prompt,
        "response": response,
    }, None


def _keep_wildchat(row: dict) -> tuple[dict | None, str | None]:
    if row.get("language") != "English":
        return None, "language"
    if row.get("redacted"):
        return None, "redacted"
    if row.get("toxic"):
        return None, "toxic"
    ex = _first_exchange(row.get("conversation") or [])
    if ex is None:
        return None, "no_single_exchange"
    prompt, response = ex
    return {
        "id": f"wildchat-{_sha(prompt + chr(30) + response)}",
        "corpus": "wildchat",
        "prompt": prompt,
        "response": response,
    }, None


def phase_found(args) -> None:
    """Stream-stage the found-response pool (bounded, checkpointed, resumable)."""
    out_dir = Path(args.out_root) / "found"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("lmsys", "lmsys/lmsys-chat-1m", _keep_lmsys),
        ("wildchat", "allenai/WildChat-1M", _keep_wildchat),
    ]
    all_rows: list[dict] = []
    for name, dataset_id, keep_fn in specs:
        fp = _fingerprint(
            issue=C.ISSUE,
            dataset=dataset_id,
            keep_cap=args.found_cap,
            stream_cap=args.found_stream_cap,
            filters="english+single_exchange+moderation_v1",
        )
        rows, counters = _stream_stage(
            out_path=out_dir / f"{name}_pool.jsonl",
            fingerprint=fp,
            row_iter_factory=lambda d=dataset_id: _hf_stream(d, None, "train"),
            keep_fn=keep_fn,
            keep_cap=args.found_cap,
            stream_cap=args.found_stream_cap,
            log_label=f"p1_found_{name}",
        )
        lib.log_phase("p1_found", f"{name}: kept={len(rows)} counters={dict(counters)}")
        all_rows.extend(rows)
    if not all_rows:
        raise RuntimeError("found pool is EMPTY — every filter rejected everything")
    # Near-dup SELF-screen over responses (MinHash-LSH), then over prompts:
    # #1739 signatures + the pool-vs-itself mask (the two-array near_dup_mask
    # is train-vs-eval and does NOT implement self-dedup — v6 crash fix).
    dup_resp = self_near_dup_mask(minhash_signatures([r["response"] for r in all_rows]))
    deduped = [r for r, d in zip(all_rows, dup_resp, strict=True) if not d]
    dup_prompt = self_near_dup_mask(minhash_signatures([r["prompt"] for r in deduped]))
    deduped = [r for r, d in zip(deduped, dup_prompt, strict=True) if not d]
    write_jsonl(out_dir / "found_pool.jsonl", deduped)
    lib.log_phase(
        "p1_found",
        f"pool: {len(all_rows)} raw -> {len(deduped)} after near-dup "
        f"(resp {int(dup_resp.sum())} + prompt {int(dup_prompt.sum())} dropped)",
    )


# ── CVEfixes (real vulnerable/fixed code + CVSS) ─────────────────────────────


def _resolve_field(row: dict, candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in row and row[k] not in (None, ""):
            return k
    return None


def _present_fields(row: dict, candidates: tuple[str, ...]) -> list[str]:
    """Candidate keys PRESENT in the row's schema (key presence, value-agnostic).

    Returned in candidate-priority order. Schema probes resolve on KEY
    presence — per-row VALUES vary (the realized CVEfixes rows carry
    cvss3_base_score, cvss2_base_score, one, both, or neither non-null), so
    value-gated resolution on the first row would mis-fire the A12 fallback
    (v12 crash fix).
    """
    return [k for k in candidates if k in row]


def _keep_cvefixes_factory() -> tuple[list, callable]:
    """Schema-probing keep_fn: resolves field names on the first row, fail loud.

    The probe fails loud (plan A12) only when the SCHEMA carries no CVSS /
    vulnerable-code candidate key at all; per-row value gaps are handled
    row-by-row — CVSS takes the first parseable candidate in priority order
    (cvss3_base_score, then cvss2_base_score, then legacy names); a row with
    no parseable score is rejected ``cvss_missing`` (counted), never the
    whole stream. The ``severity`` categorical is NOT a fallback: downstream
    banding (``band_for_cvss``) consumes a numeric score only.
    """
    resolved: dict[str, object] = {}

    def keep(row: dict) -> tuple[dict | None, str | None]:
        if not resolved:
            cvss_fields = _present_fields(row, C.CVEFIXES_CVSS_FIELDS)
            before_fields = _present_fields(row, C.CVEFIXES_CODE_BEFORE_FIELDS)
            after_fields = _present_fields(row, C.CVEFIXES_CODE_AFTER_FIELDS)
            if not cvss_fields or not before_fields:
                raise RuntimeError(
                    "CVEfixes schema probe FAILED (plan A12): row keys "
                    f"{sorted(row.keys())[:40]} resolve no CVSS/vulnerable-code field — "
                    "fall back to organic non-Qwen code completions judged insecure."
                )
            resolved.update(
                cvss=cvss_fields,
                before=before_fields[0],
                after=after_fields[0] if after_fields else "",
            )
            logger.info("[p1_cvefixes] schema resolved: %s", resolved)
        cvss = None
        for k in resolved["cvss"]:
            v = row.get(k)
            if v in (None, ""):
                continue
            try:
                cvss = float(v)
            except (TypeError, ValueError):
                continue
            break
        if cvss is None:
            return None, "cvss_missing"
        # usable_text returns the REJECT REASON, or None when usable — keep on
        # None, stage the actual field text (never the helper's return value).
        before_raw = row[resolved["before"]]
        if usable_text(before_raw, min_chars=40) is not None:
            return None, "code_before_unusable"
        before = before_raw.strip()
        after = None
        if resolved["after"]:
            after_raw = row.get(resolved["after"])
            if usable_text(after_raw, min_chars=40) is None:
                after = after_raw.strip()
        desc_f = _resolve_field(row, C.CVEFIXES_DESC_FIELDS)
        desc = None
        if desc_f:
            desc_raw = row.get(desc_f)
            if usable_text(desc_raw, min_chars=10) is None:
                desc = desc_raw.strip()
        return {
            "id": f"cvefixes-{_sha(before)}",
            "cvss": cvss,
            "desc": desc or "the following code change",
            "code_before": before,
            "code_after": after,
        }, None

    return [resolved], keep


def phase_cvefixes(args) -> None:
    """Stream-stage CVEfixes rows (CVSS + real vulnerable/fixed code)."""
    out_dir = Path(args.out_root) / "cvefixes"
    _, keep_fn = _keep_cvefixes_factory()
    fp = _fingerprint(
        issue=C.ISSUE,
        dataset=C.CVEFIXES_DATASET,
        keep_cap=args.cvefixes_cap,
        stream_cap=C.CVEFIXES_STREAM_CAP,
        # v2: realized-schema field mapping (cvss3/cvss2 per-row fallback) —
        # the mapping is a recipe constant, so a v1 partial must not resume.
        filters="cvss+code_v2",
    )
    rows, counters = _stream_stage(
        out_path=out_dir / "cvefixes_pool.jsonl",
        fingerprint=fp,
        row_iter_factory=lambda: _hf_stream(C.CVEFIXES_DATASET, None, "train"),
        keep_fn=keep_fn,
        keep_cap=args.cvefixes_cap,
        stream_cap=C.CVEFIXES_STREAM_CAP,
        log_label="p1_cvefixes",
    )
    dup = self_near_dup_mask(minhash_signatures([r["code_before"] for r in rows]))
    deduped = [r for r, d in zip(rows, dup, strict=True) if not d]
    write_jsonl(out_dir / "cvefixes_pool.jsonl", deduped)
    lib.log_phase(
        "p1_cvefixes",
        f"kept={len(deduped)} (near-dup -{len(rows) - len(deduped)}) counters={dict(counters)}",
    )


# ── LMSYS real-prompt eval panel (P5/P6 surface) ─────────────────────────────


def phase_panel_prompts(args) -> None:
    """Seeded draw of the real-prompt eval panel from the found pool."""
    import numpy as np

    pool_path = Path(args.out_root) / "found" / "found_pool.jsonl"
    rows = read_jsonl(pool_path)
    if len(rows) < C.LMSYS_PANEL_N_PROMPTS:
        raise RuntimeError(f"found pool too small for the eval panel: {len(rows)}")
    rng = np.random.default_rng(C.LMSYS_PANEL_SEED)
    idx = rng.choice(len(rows), size=C.LMSYS_PANEL_N_PROMPTS, replace=False)
    panel = [
        {"panel_idx": int(i), "id": rows[j]["id"], "prompt": rows[j]["prompt"]}
        for i, j in enumerate(sorted(idx.tolist()))
    ]
    write_jsonl(Path(args.out_root) / "panel_prompts.jsonl", panel)
    lib.log_phase(
        "p1_panel", f"eval panel staged: {len(panel)} real prompts (seed={C.LMSYS_PANEL_SEED})"
    )


# ── non-Qwen panel rollouts (vLLM) ───────────────────────────────────────────


def _generate_chunked(llm, prompts: list[str], sampling_params, *, chunk: int = 500) -> list:
    """Chunked ``LLM.generate`` (deadlock prevention, gotchas.md) with per-chunk log.

    ``sampling_params`` is a single ``SamplingParams`` (broadcast, the
    phase_rollouts path — byte-unchanged behavior) OR a list aligned 1:1 with
    ``prompts`` (the regen path: per-prompt ``n`` = that prompt's truncated
    rollout count), sliced per chunk.
    """
    per_prompt = isinstance(sampling_params, list)
    if per_prompt and len(sampling_params) != len(prompts):
        raise ValueError(f"sampling_params list {len(sampling_params)} != prompts {len(prompts)}")
    outs = []
    for lo in range(0, len(prompts), chunk):
        hi = min(lo + chunk, len(prompts))
        logger.info("[vllm-chunk] rollouts chunk %d..%d/%d", lo, hi, len(prompts))
        sp = sampling_params[lo:hi] if per_prompt else sampling_params
        outs.extend(llm.generate(prompts[lo:hi], sp, use_tqdm=False))
    return outs


def phase_rollouts(args) -> None:
    """Per-(family, panel-model) vLLM rollout sampling with cap-hit reporting."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest  # noqa: F401  (engine has enable_lora)

    out_root = Path(args.out_root)
    families = args.families or list(C.EM_FAMILIES)
    models = args.models or list(C.PANEL_MODELS)
    report: dict[str, dict] = {}
    report_path = out_root / "rollouts" / "cap_hit_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
    # Regime fingerprint: every output-affecting flag keys the resume (a re-run
    # under a different regime recomputes; #722-r3 class, review issue 8).
    fp = {
        "n_rollouts": args.n_rollouts,
        "max_new_tokens": args.max_new_tokens,
        "max_prompts": args.max_prompts,
        "temperature": C.PANEL_TEMPERATURE,
        "seed": C.PANEL_SEED,
    }
    for model_id in models:
        slug = _model_slug(model_id)
        tok = _get_tokenizer(model_id)
        pending = [
            f
            for f in families
            if not resume_ok(out_root / "rollouts" / f / f"{slug}_meta.json", fp)
        ]
        if not pending:
            lib.log_phase("p1_rollouts", f"{slug}: all families complete — skip engine build")
            continue
        attn_applied = _apply_attn_env(model_id)
        llm = None
        try:
            llm = lib.build_vllm_engine(model_id, gpu_memory_utilization=args.gpu_mem_util)
            for family in pending:
                prompts_rows = read_jsonl(out_root / "prompts" / f"{family}.jsonl")
                if args.max_prompts:
                    prompts_rows = prompts_rows[: args.max_prompts]
                budget = lib.VLLM_MAX_MODEL_LEN - args.max_new_tokens
                rendered, kept_rows, n_overlong = [], [], 0
                for r in prompts_rows:
                    text = tok.apply_chat_template(
                        [{"role": "user", "content": r["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
                    if n_tok > budget:
                        n_overlong += 1
                        continue
                    rendered.append(text)
                    kept_rows.append(r)
                sp = SamplingParams(
                    n=args.n_rollouts,
                    temperature=C.PANEL_TEMPERATURE,
                    max_tokens=args.max_new_tokens,
                    seed=C.PANEL_SEED,
                )
                outs = _generate_chunked(llm, rendered, sp)
                fam_dir = out_root / "rollouts" / family
                fam_dir.mkdir(parents=True, exist_ok=True)
                # Regime re-run (fingerprint mismatch): sweep THIS model's stale
                # shards so a smaller re-run never mixes with a prior regime's
                # higher-numbered parts (consumers glob *_part*.jsonl).
                for stale in sorted(fam_dir.glob(f"{slug}_part*.jsonl")):
                    stale.unlink()
                n_total, n_cap = 0, 0
                shard: list[dict] = []
                part = 0

                def _flush(final: bool = False) -> None:
                    nonlocal shard, part
                    if shard and (final or len(shard) >= 3000):
                        write_jsonl(fam_dir / f"{slug}_part{part:03d}.jsonl", shard)
                        part += 1
                        shard = []

                for row, out in zip(kept_rows, outs):
                    for k, comp in enumerate(out.outputs):
                        n_total += 1
                        capped = comp.finish_reason == "length"
                        n_cap += int(capped)
                        shard.append(
                            {
                                "id": f"{family}-{slug}-{row['idx']}-{k}",
                                "family": family,
                                "model": model_id,
                                "prompt_idx": row["idx"],
                                "rollout_idx": k,
                                "prompt": row["prompt"],
                                "response": comp.text,
                                "finish_reason": comp.finish_reason,
                            }
                        )
                        _flush()
                _flush(final=True)
                frac = n_cap / max(1, n_total)
                meta = {
                    "family": family,
                    "model": model_id,
                    "n_prompts": len(kept_rows),
                    "n_overlong_dropped": n_overlong,
                    "n_rollouts": n_total,
                    "cap_hit_fraction": frac,
                    "max_new_tokens": args.max_new_tokens,
                    "regen_trigger": frac > C.CAP_HIT_REGEN_THRESHOLD,
                    "reproducibility": lib.repro_metadata(),
                }
                meta_path = fam_dir / f"{slug}_meta.json"
                atomic_write_text(meta_path, json.dumps(meta, indent=2))
                write_fingerprint(meta_path, fp)
                report[f"{family}/{slug}"] = {"cap_hit_fraction": frac, "n": n_total}
                lib.log_phase(
                    "p1_rollouts",
                    f"{family}/{slug}: {n_total} rollouts, cap-hit {frac:.4f}"
                    + (
                        " REGEN-TRIGGER (>2% — re-generate at >=2x cap)"
                        if frac > C.CAP_HIT_REGEN_THRESHOLD
                        else ""
                    ),
                )
        finally:
            if llm is not None:
                lib.reap_vllm_engine(llm)
            _restore_attn_env(attn_applied)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    bad = {k: v for k, v in report.items() if v["cap_hit_fraction"] > C.CAP_HIT_REGEN_THRESHOLD}
    if bad:
        lib.log_phase(
            "p1_rollouts",
            f"CAP-HIT REGEN TRIGGER on {sorted(bad)} — run --phase rollouts_regen (>=2x cap)",
        )


# ── cap-hit regen (the >2% trigger's ACTION arm, v14) ────────────────────────


def _load_cell_shards(fam_dir: Path, slug: str) -> list[tuple[Path, list[dict]]]:
    """Read a cell's rollout shards in part order; fail loud when none exist."""
    parts = sorted(fam_dir.glob(f"{slug}_part*.jsonl"))
    if not parts:
        raise FileNotFoundError(f"no rollout shards for {slug} under {fam_dir}")
    return [(p, read_jsonl(p)) for p in parts]


def _regen_cell(llm, tok, *, fam_dir: Path, family: str, model_id: str, regen_cap: int) -> dict:
    """Re-generate ONE triggered cell's truncated rows at ``regen_cap``; splice in place.

    Selection is the persisted per-row ``finish_reason == "length"`` (written
    by ``phase_rollouts``). Row identity is stable — the same ``id`` /
    ``prompt_idx`` / ``rollout_idx`` slots get their ``response`` +
    ``finish_reason`` replaced (plus a ``regen_max_new_tokens`` provenance
    field); every other row re-serializes byte-identically, and untouched
    shard files are not rewritten. Requests are GROUPED per prompt with
    ``n =`` that prompt's truncated rollout count: two same-prompt same-seed
    n=1 requests would return IDENTICAL text (per-request seeding), while
    within-request draws differ. A prompt whose render exceeds
    ``VLLM_MAX_MODEL_LEN - regen_cap`` cannot be regenerated at the raised cap
    (gotchas.md max_model_len rule) — its rows keep the truncated text and are
    COUNTED (``regen_overlong_skipped``), never silently truncated further.
    Returns the cell's post-regen ``cap_hit_report`` entry.
    """
    from vllm import SamplingParams

    slug = _model_slug(model_id)
    meta_path = fam_dir / f"{slug}_meta.json"
    meta = json.loads(meta_path.read_text())
    orig_cap = int(meta["max_new_tokens"])
    if regen_cap < 2 * orig_cap:
        raise ValueError(
            f"{family}/{slug}: regen cap {regen_cap} < 2x the cell's original cap {orig_cap} "
            "— the pre-registered >=2x floor (#1332/#1426/#1481)"
        )
    shards = _load_cell_shards(fam_dir, slug)
    n_total = sum(len(rows) for _, rows in shards)
    if n_total != int(meta["n_rollouts"]):
        raise RuntimeError(
            f"{family}/{slug}: shard rows ({n_total}) != meta n_rollouts "
            f"({meta['n_rollouts']}) — shard/meta drift; refusing to splice"
        )
    trunc = [
        (si, ri)
        for si, (_, rows) in enumerate(shards)
        for ri, r in enumerate(rows)
        if r.get("finish_reason") == "length"
    ]
    if not trunc:
        raise RuntimeError(
            f"{family}/{slug}: regen_trigger set but zero finish_reason=='length' rows on disk"
        )
    by_prompt: dict[int, list[tuple[int, int]]] = {}
    for si, ri in trunc:
        by_prompt.setdefault(int(shards[si][1][ri]["prompt_idx"]), []).append((si, ri))
    budget = lib.VLLM_MAX_MODEL_LEN - regen_cap
    rendered: list[str] = []
    groups: list[list[tuple[int, int]]] = []
    n_overlong = 0
    for pidx in sorted(by_prompt):
        group = sorted(by_prompt[pidx], key=lambda t: int(shards[t[0]][1][t[1]]["rollout_idx"]))
        si, ri = group[0]
        text = tok.apply_chat_template(
            [{"role": "user", "content": shards[si][1][ri]["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if len(tok(text, add_special_tokens=False)["input_ids"]) > budget:
            n_overlong += len(group)
            continue
        rendered.append(text)
        groups.append(group)
    sps = [
        SamplingParams(
            n=len(group),
            temperature=C.PANEL_TEMPERATURE,
            max_tokens=regen_cap,
            seed=C.PANEL_SEED,
        )
        for group in groups
    ]
    outs = _generate_chunked(llm, rendered, sps) if rendered else []
    n_regen, n_resid = 0, 0
    changed: set[int] = set()
    for group, out in zip(groups, outs, strict=True):
        for (si, ri), comp in zip(group, out.outputs, strict=True):
            row = shards[si][1][ri]
            row["response"] = comp.text
            row["finish_reason"] = comp.finish_reason
            row["regen_max_new_tokens"] = regen_cap
            n_regen += 1
            n_resid += int(comp.finish_reason == "length")
            changed.add(si)
    for si in sorted(changed):
        path, rows = shards[si]
        write_jsonl(path, rows)
    post_cap_hits = sum(1 for _, rows in shards for r in rows if r.get("finish_reason") == "length")
    post_frac = post_cap_hits / max(1, n_total)
    resid_frac = n_resid / max(1, n_regen)
    pre_frac = float(meta["cap_hit_fraction"])
    meta.update(
        regen_applied=True,
        regen_max_new_tokens=regen_cap,
        regen_n_rows=n_regen,
        regen_overlong_skipped=n_overlong,
        residual_after_regen=resid_frac,
        cap_hit_fraction_pre_regen=pre_frac,
        cap_hit_fraction=post_frac,
        regen_reproducibility=lib.repro_metadata(),
    )
    atomic_write_text(meta_path, json.dumps(meta, indent=2))
    lib.log_phase(
        "p1_rollouts_regen",
        f"{family}/{slug}: regenerated n={n_regen} rows at {regen_cap}, "
        f"residual cap-hit {resid_frac:.4f} "
        f"(cell {pre_frac:.4f} -> {post_frac:.4f}; overlong-skipped {n_overlong})",
    )
    if resid_frac > C.CAP_HIT_REGEN_THRESHOLD:
        lib.log_phase(
            "p1_rollouts_regen",
            f"{family}/{slug}: RESIDUAL cap-hit {resid_frac:.4f} > "
            f"{C.CAP_HIT_REGEN_THRESHOLD} at cap {regen_cap} — digest caveat "
            "(one regen pass only; no unbounded cap iteration)",
        )
    return {
        "cap_hit_fraction": post_frac,
        "n": n_total,
        "regen_applied": True,
        "pre_regen_cap_hit_fraction": pre_frac,
        "residual_after_regen": resid_frac,
        "regen_n_rows": n_regen,
        "regen_max_new_tokens": regen_cap,
    }


def phase_rollouts_regen(args) -> None:
    """Cap-hit regen ACTION arm: re-generate triggered cells' truncated rows.

    The pre-registered rule (CLAUDE.md ``max_new_tokens``; #1332/#1426/#1481):
    cap-hit > 2% per (family, model) cell => re-generate THOSE rows at >= 2x
    the cap. ``phase_rollouts`` computes + logs the trigger; THIS phase acts
    on it BEFORE P2 banding consumes the rollouts (v14 — the trigger was
    report-only through v13). One regen pass only: a residual cap-hit at the
    regen cap is reported (meta ``residual_after_regen`` + cap_hit_report),
    never iterated. Idempotent via meta ``regen_applied``; the meta's base
    ``.fp.json`` sidecar is untouched (same rollouts regime keys), so a
    ``--phase rollouts`` re-run still resume-skips. Preserves the per-model
    attention-kernel pins (gemma FA2, v13) and the per-model fan-out contract
    (``--models <one>`` under a launcher CVD pin).
    """
    out_root = Path(args.out_root)
    families = args.families or list(C.EM_FAMILIES)
    models = args.models or list(C.PANEL_MODELS)
    regen_cap = (
        args.regen_max_new_tokens
        if args.regen_max_new_tokens is not None
        else 2 * args.max_new_tokens
    )
    report_path = out_root / "rollouts" / "cap_hit_report.json"
    report: dict[str, dict] = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text())
    for model_id in models:
        slug = _model_slug(model_id)
        pending: list[str] = []
        for family in families:
            meta_path = out_root / "rollouts" / family / f"{slug}_meta.json"
            if not meta_path.is_file():
                raise FileNotFoundError(
                    f"rollouts meta missing for {family}/{slug} ({meta_path}) — "
                    "run --phase rollouts first"
                )
            meta = json.loads(meta_path.read_text())
            if meta.get("regen_applied"):
                lib.log_phase("p1_rollouts_regen", f"{family}/{slug}: regen already applied — skip")
                continue
            if not meta.get("regen_trigger"):
                lib.log_phase(
                    "p1_rollouts_regen",
                    f"{family}/{slug}: no regen trigger "
                    f"(cap-hit {meta['cap_hit_fraction']:.4f}) — skip",
                )
                continue
            # Validate the >=2x floor BEFORE any engine build (fail fast on a
            # mis-passed flag); _regen_cell re-checks (defense in depth).
            if regen_cap < 2 * int(meta["max_new_tokens"]):
                raise ValueError(
                    f"{family}/{slug}: regen cap {regen_cap} < 2x the cell's original cap "
                    f"{meta['max_new_tokens']} — the pre-registered >=2x floor (#1332/#1426/#1481)"
                )
            pending.append(family)
        if not pending:
            lib.log_phase("p1_rollouts_regen", f"{slug}: no triggered cells — skip engine build")
            continue
        tok = _get_tokenizer(model_id)
        attn_applied = _apply_attn_env(model_id)
        llm = None
        try:
            llm = lib.build_vllm_engine(model_id, gpu_memory_utilization=args.gpu_mem_util)
            for family in pending:
                entry = _regen_cell(
                    llm,
                    tok,
                    fam_dir=out_root / "rollouts" / family,
                    family=family,
                    model_id=model_id,
                    regen_cap=regen_cap,
                )
                report[f"{family}/{slug}"] = entry
                # Checkpoint the report per CELL (never write-at-end).
                report_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(report_path, json.dumps(report, indent=2))
        finally:
            if llm is not None:
                lib.reap_vllm_engine(llm)
            _restore_attn_env(attn_applied)
    bad = {k: v for k, v in report.items() if v["cap_hit_fraction"] > C.CAP_HIT_REGEN_THRESHOLD}
    if bad:
        lib.log_phase(
            "p1_rollouts_regen",
            f"post-regen cap-hit still > {C.CAP_HIT_REGEN_THRESHOLD} on {sorted(bad)} — "
            "digest caveat (one regen pass; no unbounded cap iteration)",
        )


# ── upload ────────────────────────────────────────────────────────────────────


def phase_upload(args) -> None:
    """Persist ALL staged text to the HF data repo (batched folder commits)."""
    from explore_persona_space.orchestrate import hub

    out_root = Path(args.out_root)
    mapping = {
        "prompts": f"{C.HF_PREFIX}/raw_completions/prompts",
        "rollouts": f"{C.HF_PREFIX}/raw_completions/rollouts",
        "found": f"{C.HF_PREFIX}/raw_completions/found",
        "cvefixes": f"{C.HF_PREFIX}/raw_completions/found/cvefixes",
    }
    for sub, prefix in mapping.items():
        local = out_root / sub
        if not local.is_dir():
            logger.info("[p1_upload] %s absent — skip", local)
            continue
        url = hub._upload(local, C.HF_DATA_REPO, "dataset", prefix, raise_on_error=True)
        lib.log_phase("p1_upload", f"{sub} -> {url}")
    panel = out_root / "panel_prompts.jsonl"
    if panel.is_file():
        # UPLOAD_RETURN_DISCARD_EXEMPT: raise_on_error=True — failure raises, URL unused
        hub._upload(
            panel,
            C.HF_DATA_REPO,
            "dataset",
            f"{C.HF_PREFIX}/raw_completions/panel_prompts.jsonl",
            upload_as_file=True,
            raise_on_error=True,
        )


PHASES = {
    "prompts": phase_prompts,
    "found": phase_found,
    "cvefixes": phase_cvefixes,
    "panel_prompts": phase_panel_prompts,
    "rollouts": phase_rollouts,
    # regen sits BETWEEN rollouts and upload so `--phase all` is the unified
    # self-healing pipeline: triggered cells are regenerated before upload.
    "rollouts_regen": phase_rollouts_regen,
    "upload": phase_upload,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    ap.add_argument("--out-root", default="data/issue_2221/corpus")
    ap.add_argument("--external-root", default="external/persona_vectors")
    ap.add_argument("--families", nargs="*", default=None, help="EM families subset (rollouts)")
    ap.add_argument("--models", nargs="*", default=None, help="panel model subset (rollouts)")
    ap.add_argument("--prompts-cap", type=int, default=C.EM_PROMPTS_CAP_PER_FAMILY)
    ap.add_argument("--max-prompts", type=int, default=None, help="smoke: cap prompts per family")
    ap.add_argument("--n-rollouts", type=int, default=C.N_PANEL_ROLLOUTS)
    ap.add_argument("--max-new-tokens", type=int, default=C.PANEL_MAX_NEW_TOKENS)
    ap.add_argument(
        "--regen-max-new-tokens",
        type=int,
        default=None,
        help="rollouts_regen cap; default 2x --max-new-tokens (the >=2x rule floor)",
    )
    ap.add_argument("--found-cap", type=int, default=C.FOUND_KEEP_CAP_PER_CORPUS)
    ap.add_argument("--found-stream-cap", type=int, default=C.FOUND_STREAM_CAP)
    ap.add_argument("--cvefixes-cap", type=int, default=C.CVEFIXES_KEEP_CAP)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        # Execute the deferred imports (smoke-architecture Axis 1).
        from vllm import LLM, SamplingParams  # noqa: F401
        from vllm.lora.request import LoRARequest  # noqa: F401

        from explore_persona_space.orchestrate import hub  # noqa: F401

        print("[import-check] OK")
        raise SystemExit(0)
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    for name in phases:
        lib.log_phase(f"p1_{name}", "start")
        PHASES[name](args)
    lib.log_phase("p1", "done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
