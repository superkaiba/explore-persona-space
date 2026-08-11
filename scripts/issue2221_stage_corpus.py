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
    """Chunked ``LLM.generate`` (deadlock prevention, gotchas.md) with per-chunk log."""
    outs = []
    for lo in range(0, len(prompts), chunk):
        hi = min(lo + chunk, len(prompts))
        logger.info("[vllm-chunk] rollouts chunk %d..%d/%d", lo, hi, len(prompts))
        outs.extend(llm.generate(prompts[lo:hi], sampling_params, use_tqdm=False))
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
        llm = lib.build_vllm_engine(model_id, gpu_memory_utilization=args.gpu_mem_util)
        try:
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
            lib.reap_vllm_engine(llm)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    bad = {k: v for k, v in report.items() if v["cap_hit_fraction"] > C.CAP_HIT_REGEN_THRESHOLD}
    if bad:
        lib.log_phase("p1_rollouts", f"CAP-HIT REGEN TRIGGER on {sorted(bad)} — rerun with 2x cap")


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
