"""Issue #2221 P5 — activation capture (last-prompt-token + prefix-end + response-avg).

Phases (``--phase``; registry ``PHASES``):

- ``surfaces`` : freeze the capture surface — the paper's 20 held-out eval
                 questions per trait (60) + the 50-prompt LMSYS real panel.
- ``parity``   : apply-and-read parity probe (BEFORE any synthetic-stratum
                 compute): apply the reused #778 adapter ``evil_misaligned_2``
                 on the CURRENT stack, reproduce its cached last-prompt-token
                 shift projection (``finetune_activations/``) within tolerance —
                 a mismatch HALTS the leg (artifact-reuse check).
- ``last``     : last-prompt-token AND prefix-end states captured in the SAME
                 forwards, all 28 layers, for base + 24 real-twin finals + 72
                 frac-checkpoints. The prefix-end position (last chat-template
                 header token BEFORE the user query) is derived from OFFSET
                 MAPPINGS (BPE-seam safe, gotchas.md).
- ``gen``      : greedy on-policy responses (vLLM, shared enable_lora engine,
                 one LoRARequest per adapter) for base + 24 twins + 24 reused
                 #778 adapters — persisted BEFORE any capture reduce.
- ``resp``     : response-avg capture (``issue778_lib.capture_response_avg_all_layers``,
                 the reused #778 code path) for base + 24 twins + 24 synth778.
- ``upload``   : capture stores -> HF, rooted at ``--upload-prefix`` (default
                 the parent round's ``analysis_tensors`` -> ``issue2221_realtwin/
                 analysis_tensors/{capture,capture_resp,...}``). The
                 specialized_corpus_remine P5 invocation passes
                 ``--upload-prefix analysis_tensors/remine_capture`` so remine
                 stores can NEVER resolve to (and clobber) a parent prefix
                 (concern capture-upload-prefix-remine-clobber; plan v11 §4).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import os  # noqa: E402

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # gotchas.md #628

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue778_lib as lib  # noqa: E402

from explore_persona_space.experiments.issue_2221 import constants as C  # noqa: E402
from explore_persona_space.experiments.issue_2221.loaders import (  # noqa: E402
    atomic_torch_save,
    atomic_write_text,
    load_ft_activation,
    load_rb,
    read_jsonl,
    resume_ok,
    write_fingerprint,
)

logger = logging.getLogger("issue2221.capture")

PARITY_MIN_PROFILE_COS = 0.99  # structural breakage reads ~0; bf16 stack jitter << 0.01
# HF sub-roots for phase_upload (plan v11 §4): default = the parent round's
# hardcoded value (byte-identical parent behavior); the remine P5 run passes
# REMINE_UPLOAD_PREFIX so its stores can never clobber a parent prefix.
PARENT_UPLOAD_PREFIX = "analysis_tensors"
REMINE_UPLOAD_PREFIX = "analysis_tensors/remine_capture"
# The sentinel only locates the CHAR offset where user content starts in the
# fixed chat template (content-independent), so a plain ASCII token suffices.
_SENTINEL_QUERY = "EPMQ2221SENTINELQ"


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(lib.MODEL_NAME)


def _prefix_char_len(tok) -> int:
    """Chars before the user-query content in the rendered chat template."""
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": _SENTINEL_QUERY}], tokenize=False, add_generation_prompt=True
    )
    pos = rendered.find(_SENTINEL_QUERY)
    assert pos > 0, "sentinel not found in rendered chat template"
    return pos


def prefix_end_index(offsets: list[tuple[int, int]], prefix_char_len: int) -> int:
    """Last token ENDING inside the prefix text (offset-mapping convention)."""
    idx = -1
    for i, (_, end) in enumerate(offsets):
        if 0 < end <= prefix_char_len:
            idx = i
    assert idx >= 1, f"degenerate prefix-end index {idx} (prefix_char_len={prefix_char_len})"
    return idx


def capture_last_and_prefix(model, tok, prompts: list[str], *, device) -> tuple:
    """(last, prefix) hidden-state summaries, each (n, 28, 3584) fp16, ONE forward per prompt.

    Layer convention matches ``issue778_lib``: stored ``[layer_idx]`` is
    ``hidden_states[layer_idx + 1]`` (the output of block ``layer_idx + 1``).
    """
    import torch

    from explore_persona_space.analysis.extraction import _logits_to_keep_kwargs

    pcl = _prefix_char_len(tok)
    last_rows, prefix_rows = [], []
    # return_logits=False: this capture reads ONLY hook-captured hidden states —
    # the helper's memory-optimal contract (skips the B x T x vocab lm_head
    # materialization; extraction.py docstring). The kwarg is REQUIRED (v8 fix:
    # the bare 1-arg call crashed capture:parity on-pod; --import-check resolves
    # the import but never binds the call).
    lk = _logits_to_keep_kwargs(model, return_logits=False)
    with torch.no_grad():
        for text in prompts:
            enc = tok(
                text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False
            )
            offsets = [tuple(x) for x in enc.pop("offset_mapping")[0].tolist()]
            p_idx = prefix_end_index(offsets, pcl)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, **lk)
            hs = out.hidden_states  # tuple(len 29) of (1, T, d)
            assert len(hs) == C.N_LAYERS + 1, len(hs)
            last = torch.stack([hs[layer + 1][0, -1, :] for layer in range(C.N_LAYERS)])
            pref = torch.stack([hs[layer + 1][0, p_idx, :] for layer in range(C.N_LAYERS)])
            assert last.shape == (C.N_LAYERS, C.HIDDEN_DIM), last.shape
            last_rows.append(last.to(torch.float16).cpu())
            prefix_rows.append(pref.to(torch.float16).cpu())
    import torch as _t

    return _t.stack(last_rows), _t.stack(prefix_rows)


# ── model roster ──────────────────────────────────────────────────────────────


def all_cells() -> list[str]:
    return [f"{f}_{v}" for f in C.FAMILIES for v in C.VERSIONS]


def stage_synth_adapter(cell: str, stage_dir: Path) -> Path:
    """Stage one reused #778 adapter from the model repo @ the plan pin."""
    from explore_persona_space.orchestrate import hub

    prefix = f"{C.ADAPTERS_778_PREFIX}/{cell}"
    # stage_hub_prefix mirrors the repo-relative path under dest_dir (#1774).
    local = stage_dir / prefix
    if not (local / "adapter_config.json").is_file():
        hub.stage_hub_prefix(
            C.HF_MODEL_REPO,
            prefix,
            stage_dir,
            repo_type="model",
            revision=C.ADAPTERS_778_REVISION,
        )
    if not (local / "adapter_config.json").is_file():
        raise FileNotFoundError(f"staged #778 adapter incomplete: {local}")
    return local


def model_roster(args) -> list[tuple[str, Path | None]]:
    """(model_tag, adapter_dir|None) rows for the last-token capture leg."""
    ckpt_root = Path(args.ckpt_root)
    roster: list[tuple[str, Path | None]] = [("base", None)]
    cells = args.cells or all_cells()
    for cell in cells:
        adapter = ckpt_root / cell
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"real-twin adapter missing: {adapter}")
        roster.append((cell, adapter))
        for frac in C.CHECKPOINT_FRACS:
            ck = adapter / f"checkpoint_frac{int(round(frac * 100))}"
            if (ck / "adapter_config.json").is_file():
                roster.append((f"{cell}@frac{int(round(frac * 100))}", ck))
            else:
                logger.warning("[p5] checkpoint missing (named, not zero-barred): %s", ck)
    return roster


# ── phases ────────────────────────────────────────────────────────────────────


def phase_surfaces(args) -> None:
    """Freeze the capture surface (paper 20-q per trait + the LMSYS panel)."""
    rows: list[dict] = []
    for trait in lib.TRAITS:
        td = lib.load_trait_data(Path(args.external_root), trait)
        for i, q in enumerate(td.eval_questions[: args.n_questions]):
            rows.append(
                {
                    "surface_id": f"paper-{trait}-{i:02d}",
                    "kind": "paper",
                    "trait": trait,
                    "prompt": q,
                }
            )
    panel_path = Path(args.corpus_root) / "panel_prompts.jsonl"
    if panel_path.is_file():
        for r in read_jsonl(panel_path):
            rows.append(
                {
                    "surface_id": f"lmsys-{r['panel_idx']:03d}",
                    "kind": "lmsys",
                    "trait": None,
                    "prompt": r["prompt"],
                }
            )
    else:
        logger.warning("[p5] panel prompts missing at %s — paper surface only", panel_path)
    out = Path(args.out_root) / "capture_surfaces.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "reproducibility": lib.repro_metadata()}, indent=2))
    lib.log_phase("p5_surfaces", f"{len(rows)} capture prompts frozen")


def _load_surfaces(args) -> list[dict]:
    p = Path(args.out_root) / "capture_surfaces.json"
    if not p.is_file():
        raise FileNotFoundError(f"run --phase surfaces first: {p}")
    return json.loads(p.read_text())["rows"]


def _surface_roster_sha(surfaces: list[dict]) -> str:
    """CONTENT hash of the frozen surface roster (fingerprint chaining, N5).

    Every capture phase's resume fingerprint carries this hash — not just
    ``n_questions`` — so a RE-FROZEN roster (same count, different prompts /
    ids / order) invalidates cached captures instead of silently reusing
    tensors computed on stale surfaces.
    """
    from explore_persona_space.experiments.issue_2221.loaders import sha256_text

    return sha256_text(json.dumps(surfaces, sort_keys=True))


def _load_base_model():
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(lib.MODEL_NAME, torch_dtype=dtype)
    return model.to(device), device


def phase_parity(args) -> None:
    """Apply-and-read parity probe on the reused #778 evil_misaligned_2 adapter."""
    import numpy as np
    import torch
    from peft import PeftModel

    tok = _tokenizer()
    td = lib.load_trait_data(Path(args.external_root), "evil")
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        for q in td.eval_questions[: args.n_questions]
    ]
    stage_dir = Path(args.stage_dir)
    adapter_dir = stage_synth_adapter("evil_misaligned_2", stage_dir)

    model, device = _load_base_model()
    base_last, _ = capture_last_and_prefix(model, tok, prompts, device=device)
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
    cell_last, _ = capture_last_and_prefix(peft_model, tok, prompts, device=device)
    model = peft_model.unload()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    mine_base = base_last.to(torch.float64).mean(dim=0).numpy()  # (28, 3584)
    mine_cell = cell_last.to(torch.float64).mean(dim=0).numpy()
    rb = load_rb("evil", stage_dir=stage_dir)
    proj_mine = np.einsum("ld,ld->l", rb, mine_cell - mine_base)

    cached_base = load_ft_activation("base", stage_dir=stage_dir)["evil"]
    cached_cell = load_ft_activation("evil_misaligned_2", stage_dir=stage_dir)["evil"]
    proj_cached = np.einsum("ld,ld->l", rb, cached_cell - cached_base)

    cos = float(
        np.dot(proj_mine, proj_cached) / (np.linalg.norm(proj_mine) * np.linalg.norm(proj_cached))
    )
    argmax_cached = int(np.argmax(np.abs(proj_cached)))
    sign_ok = bool(np.sign(proj_mine[argmax_cached]) == np.sign(proj_cached[argmax_cached]))
    verdict = {
        "profile_cosine": cos,
        "min_profile_cosine": PARITY_MIN_PROFILE_COS,
        "argmax_layer_cached": argmax_cached,
        "sign_match_at_argmax": sign_ok,
        "passed": bool(cos >= PARITY_MIN_PROFILE_COS and sign_ok),
        "reproducibility": lib.repro_metadata(),
    }
    out = Path(args.out_root) / "parity_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    if not verdict["passed"]:
        raise RuntimeError(
            f"P5 apply-and-read PARITY PROBE FAILED: profile cos={cos:.4f} "
            f"(floor {PARITY_MIN_PROFILE_COS}), sign_match={sign_ok} — the reused #778 "
            "adapter/capture path does not reproduce the cached shift; HALT the "
            "synthetic-stratum leg (artifact-reuse check (e))."
        )
    lib.log_phase("p5_parity", f"PASS cos={cos:.4f} sign_match={sign_ok}")


def phase_last(args) -> None:
    """Last-prompt-token + prefix-end capture for base + finals + checkpoints."""
    from peft import PeftModel

    tok = _tokenizer()
    surfaces = _load_surfaces(args)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in surfaces
    ]
    surface_ids = [r["surface_id"] for r in surfaces]
    cap_dir = Path(args.out_root) / "capture"
    cap_dir.mkdir(parents=True, exist_ok=True)
    # `capture_kind` is load-bearing in the regime key, not decoration (r18):
    # without it a response-avg store and a last-token store for the same tag
    # are fingerprint-IDENTICAL, so each silently satisfies the other's
    # `resume_ok` — which is why the r18 prefix collision overwrote a whole
    # capture class without any gate firing.
    fp = {
        "n_questions": args.n_questions,
        "surfaces_sha256": _surface_roster_sha(surfaces),
        "capture_kind": "last_prompt_token+prefix_end",
    }
    pending = [
        (tag, adapter)
        for tag, adapter in model_roster(args)
        if not resume_ok(cap_dir / f"{tag}.pt", fp)
    ]
    if not pending:
        lib.log_phase("p5_last", "all captures present (fingerprint match) — skip model load")
        return
    model, device = _load_base_model()
    for tag, adapter in pending:
        dest = cap_dir / f"{tag}.pt"
        if adapter is None:
            last, pref = capture_last_and_prefix(model, tok, prompts, device=device)
        else:
            peft_model = PeftModel.from_pretrained(model, str(adapter))
            last, pref = capture_last_and_prefix(peft_model, tok, prompts, device=device)
            model = peft_model.unload()
        atomic_torch_save(
            dest,
            {
                "kind": "last_prompt_token+prefix_end",
                "last": last,
                "prefix": pref,
                "surface_ids": surface_ids,
                "model_tag": tag,
                "reproducibility": lib.repro_metadata(),
            },
        )
        write_fingerprint(dest, fp)
        lib.log_phase("p5_last", f"{tag}: captured {tuple(last.shape)}")


def phase_gen(args) -> None:
    """Greedy on-policy responses for the response-avg leg (persist text FIRST)."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tok = _tokenizer()
    surfaces = _load_surfaces(args)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        for r in surfaces
    ]
    resp_dir = Path(args.out_root) / "capture_responses"
    resp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_root)
    stage_dir = Path(args.stage_dir)
    cells = args.cells or all_cells()
    roster: list[tuple[str, Path | None]] = [("base", None)]
    roster += [(cell, ckpt_root / cell) for cell in cells]
    if not args.skip_synth:
        roster += [(f"synth778_{cell}", stage_synth_adapter(cell, stage_dir)) for cell in cells]

    fp = {
        "n_questions": args.n_questions,
        "max_new_tokens": lib.MAX_NEW_TOKENS,
        "surfaces_sha256": _surface_roster_sha(surfaces),
    }
    llm = lib.build_vllm_engine(gpu_memory_utilization=args.gpu_mem_util)
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=lib.MAX_NEW_TOKENS)
        for i, (tag, adapter) in enumerate(roster):
            dest = resp_dir / f"{tag}.json"
            if resume_ok(dest, fp):
                continue
            lora = LoRARequest(tag, i + 1, str(adapter)) if adapter is not None else None
            outs = []
            for lo in range(0, len(prompts), 500):
                logger.info("[vllm-chunk] gen %s chunk %d/%d", tag, lo, len(prompts))
                outs.extend(
                    llm.generate(prompts[lo : lo + 500], sp, lora_request=lora, use_tqdm=False)
                )
            rows = [
                {
                    "surface_id": r["surface_id"],
                    "prompt": r["prompt"],
                    "response": o.outputs[0].text,
                    "finish_reason": o.outputs[0].finish_reason,
                }
                for r, o in zip(surfaces, outs)
            ]
            cap_hit = sum(1 for r in rows if r["finish_reason"] == "length") / max(1, len(rows))
            atomic_write_text(
                dest,
                json.dumps({"rows": rows, "cap_hit_fraction": cap_hit, "model_tag": tag}, indent=2),
            )
            write_fingerprint(dest, fp)
            lib.log_phase("p5_gen", f"{tag}: {len(rows)} greedy responses (cap-hit {cap_hit:.3f})")
    finally:
        lib.reap_vllm_engine(llm)


def phase_resp(args) -> None:
    """Response-avg capture (reused #778 code path) for base + twins + synth778."""
    import torch
    from peft import PeftModel

    tok = _tokenizer()
    surfaces = _load_surfaces(args)
    resp_dir = Path(args.out_root) / "capture_responses"
    ckpt_root = Path(args.ckpt_root)
    stage_dir = Path(args.stage_dir)
    cells = args.cells or all_cells()
    roster: list[tuple[str, Path | None, Path]] = [
        ("base", None, Path(args.out_root) / "capture_resp")
    ]
    roster += [(cell, ckpt_root / cell, Path(args.out_root) / "capture_resp") for cell in cells]
    if not args.skip_synth:
        roster += [
            (
                f"synth778_{cell}",
                stage_synth_adapter(cell, stage_dir),
                Path(args.out_root) / "capture_resp_synth778",
            )
            for cell in cells
        ]
    # See phase_last's note: `capture_kind` keeps a response-avg store from
    # being fingerprint-indistinguishable from a last-token store (r18).
    fp = {
        "n_questions": args.n_questions,
        "surfaces_sha256": _surface_roster_sha(surfaces),
        "capture_kind": "response_avg",
    }
    # Roster-wide gen-output existence sweep BEFORE the 7B model load (review
    # issue 7): a missing rollout file must fail in seconds, not after the
    # weights are resident.
    pending = [
        (tag, adapter, out_dir)
        for tag, adapter, out_dir in roster
        if not resume_ok(out_dir / f"{tag}.pt", fp)
    ]
    missing = [
        str(resp_dir / f"{tag}.json")
        for tag, _, _ in pending
        if not (resp_dir / f"{tag}.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"run --phase gen first — {len(missing)} rollout file(s) missing, e.g. {missing[:3]}"
        )
    if not pending:
        lib.log_phase("p5_resp", "all response-avg captures present — skip model load")
        return
    model, device = _load_base_model()
    for tag, adapter, out_dir in pending:
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{tag}.pt"
        resp_path = resp_dir / f"{tag}.json"
        rows = json.loads(resp_path.read_text())["rows"]
        by_id = {r["surface_id"]: r for r in rows}
        prompts, responses, kept_ids = [], [], []
        for s in surfaces:
            r = by_id.get(s["surface_id"])
            if r is None or not r["response"].strip():
                continue
            prompts.append(
                tok.apply_chat_template(
                    [{"role": "user", "content": s["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            responses.append(r["response"])
            kept_ids.append(s["surface_id"])
        if adapter is None:
            acts = lib.capture_response_avg_all_layers(
                model, tok, prompts, responses, device=device
            )
        else:
            peft_model = PeftModel.from_pretrained(model, str(adapter))
            acts = lib.capture_response_avg_all_layers(
                peft_model, tok, prompts, responses, device=device
            )
            model = peft_model.unload()
        assert acts.shape == (len(prompts), C.N_LAYERS, C.HIDDEN_DIM), acts.shape
        atomic_torch_save(
            dest,
            {
                "kind": "response_avg",
                "resp_avg": acts.to(torch.float16),
                "surface_ids": kept_ids,
                "model_tag": tag,
                "reproducibility": lib.repro_metadata(),
            },
        )
        write_fingerprint(dest, fp)
        lib.log_phase("p5_resp", f"{tag}: response-avg captured {tuple(acts.shape)}")


_UPLOAD_SUBDIRS = ("capture", "capture_resp", "capture_resp_synth778", "capture_responses")
_UPLOAD_EXTRAS = ("capture_surfaces.json", "parity_probe.json")


def upload_mapping(upload_prefix: str = PARENT_UPLOAD_PREFIX) -> dict[str, str]:
    """Local-name -> HF-prefix map for ``phase_upload``, rooted at ``upload_prefix``.

    Keys are the local capture SUBDIR names (``_UPLOAD_SUBDIRS``) plus the two
    metadata FILES (``_UPLOAD_EXTRAS``). The default (``analysis_tensors``, the
    parent round's value) reproduces the parent destinations byte-identically;
    the specialized_corpus_remine P5 invocation passes
    ``analysis_tensors/remine_capture`` (``REMINE_UPLOAD_PREFIX``) so every
    destination — the ``raw_completions`` responses prefix included — lands
    under a ``remine_capture/`` root, and a structural assert guarantees a
    non-parent prefix can NEVER resolve to a parent destination (concern
    capture-upload-prefix-remine-clobber; plan v11 §4/§10).

    Layout under the root stays ``{kind-dir}/{tag}.pt`` — the prefix names
    MIRROR the local subdir names exactly, which is what
    ``issue2221_monitors._load_capture`` resolves ({"last": "capture",
    "resp": "capture_resp", "resp_synth": "capture_resp_synth778"}).
    """
    prefix = upload_prefix.strip("/")
    if not prefix:
        raise ValueError("--upload-prefix must be a non-empty HF sub-root")
    # The rollout-text store lives under raw_completions/ (upload-policy row),
    # not under the tensors root; a non-parent invocation nests it under the
    # prefix's leaf (e.g. raw_completions/remine_capture/capture_responses) so
    # remine responses never overwrite the parent's (tags collide by design:
    # remine cell ids reuse the parent's `{family}_{version}` naming).
    leaf = prefix.removeprefix(f"{PARENT_UPLOAD_PREFIX}/")
    responses_root = (
        f"{C.HF_PREFIX}/raw_completions"
        if prefix == PARENT_UPLOAD_PREFIX
        else f"{C.HF_PREFIX}/raw_completions/{leaf}"
    )
    mapping = {
        "capture": f"{C.HF_PREFIX}/{prefix}/capture",
        "capture_resp": f"{C.HF_PREFIX}/{prefix}/capture_resp",
        "capture_resp_synth778": f"{C.HF_PREFIX}/{prefix}/capture_resp_synth778",
        "capture_responses": f"{responses_root}/capture_responses",
        "capture_surfaces.json": f"{C.HF_PREFIX}/{prefix}/capture_surfaces.json",
        "parity_probe.json": f"{C.HF_PREFIX}/{prefix}/parity_probe.json",
    }
    assert len(set(mapping.values())) == len(mapping), (
        "two local capture subdirs share one HF prefix — identical filenames "
        f"would overwrite each other: {mapping}"
    )
    if prefix != PARENT_UPLOAD_PREFIX:
        clobber = set(mapping.values()) & set(upload_mapping(PARENT_UPLOAD_PREFIX).values())
        assert not clobber, (
            f"non-parent --upload-prefix {upload_prefix!r} resolves to parent "
            f"destinations — remine would clobber the parent round: {sorted(clobber)}"
        )
    return mapping


def phase_upload(args) -> None:
    """Capture stores -> HF data repo (batched folder commits).

    Destinations come from ``upload_mapping(args.upload_prefix)`` — ONE PREFIX
    PER LOCAL SUBDIR. #2221 r18 history: this mapping previously sent BOTH
    `capture` (last-prompt-token + prefix-end states) AND `capture_resp`
    (response-avg states) to the SAME `analysis_tensors/capture` prefix. The
    two stores are DISTINCT quantities whose filenames are identical — the tag
    is the filename and the KIND lives only in the directory name — so the
    second upload OVERWROTE the first file-for-file. Realized damage: the
    last-token/prefix captures for base + all 24 cells were destroyed on the
    Hub (only the 54 frac-tag stores, absent from the resp roster, survived),
    and the pod holding the local originals had already been torn down. It was
    silent because the fingerprint sidecars omitted the capture KIND (fixed in
    r18), so a resp store satisfied a last store's `resume_ok`.

    `capture_resp_synth778` also used to be RENAMED to `capture_synth778` on
    upload, which no consumer subdir matches; it now mirrors its local name.
    The legacy `analysis_tensors/capture_synth778` prefix still holds that
    class under the old name and is left in place for older readers.
    """
    from explore_persona_space.orchestrate import hub

    out_root = Path(args.out_root)
    mapping = upload_mapping(args.upload_prefix)
    for sub in _UPLOAD_SUBDIRS:
        local = out_root / sub
        if not local.is_dir():
            logger.info("[p5_upload] %s absent — skip", local)
            continue
        url = hub._upload(local, C.HF_DATA_REPO, "dataset", mapping[sub], raise_on_error=True)
        lib.log_phase("p5_upload", f"{sub} -> {url}")
    for extra in _UPLOAD_EXTRAS:
        p = out_root / extra
        if p.is_file():
            # UPLOAD_LOOP_EXEMPT: bounded fixed 2-file metadata list — never a per-cell storm
            hub._upload(  # UPLOAD_RETURN_DISCARD_EXEMPT: raise_on_error=True — failure raises; URL unused
                p,
                C.HF_DATA_REPO,
                "dataset",
                mapping[extra],
                upload_as_file=True,
                raise_on_error=True,
            )


PHASES = {
    "surfaces": phase_surfaces,
    "parity": phase_parity,
    "last": phase_last,
    "gen": phase_gen,
    "resp": phase_resp,
    "upload": phase_upload,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    ap.add_argument("--out-root", default="data/issue_2221/p5")
    ap.add_argument("--corpus-root", default="data/issue_2221/corpus")
    ap.add_argument("--ckpt-root", default="checkpoints/issue_2221")
    ap.add_argument("--stage-dir", default="data/issue_2221/hf_dl")
    ap.add_argument("--external-root", default="external/persona_vectors")
    ap.add_argument("--cells", nargs="*", default=None, help="cell subset 'family_version'")
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--skip-synth", action="store_true", help="skip the synth778 stratum")
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    ap.add_argument(
        "--upload-prefix",
        default=PARENT_UPLOAD_PREFIX,
        help="HF sub-root for phase_upload destinations under "
        f"{C.HF_PREFIX}/ (default: the parent round's {PARENT_UPLOAD_PREFIX!r}; "
        f"the specialized_corpus_remine P5 run passes {REMINE_UPLOAD_PREFIX!r} "
        "so remine stores never clobber parent prefixes — plan v11 §4)",
    )
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
        from peft import PeftModel  # noqa: F401
        from vllm import SamplingParams  # noqa: F401
        from vllm.lora.request import LoRARequest  # noqa: F401

        from explore_persona_space.analysis.extraction import (  # noqa: F401
            _logits_to_keep_kwargs,
        )
        from explore_persona_space.orchestrate import hub  # noqa: F401

        print("[import-check] OK")
        raise SystemExit(0)
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    # The parity probe MUST precede any synthetic-stratum compute (plan §4 P5).
    if "gen" in phases and not args.skip_synth:
        parity_out = Path(args.out_root) / "parity_probe.json"
        if not parity_out.is_file() and "parity" not in phases:
            raise RuntimeError("synthetic-stratum gen requires the parity probe first")
    for name in phases:
        lib.log_phase(f"p5_{name}", "start")
        PHASES[name](args)
    lib.log_phase("p5", "done")
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
