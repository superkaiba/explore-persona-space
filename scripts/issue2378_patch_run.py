#!/usr/bin/env python3
"""Issue #2378 `causal-patching-arms` pod driver (same-issue follow-up round).

Cross-framing context-end v_C patching on Qwen3.6-27B. Method + gates:
``scripts/issue2378_patch_common.py`` (the round's constants + screen rule);
estimator/DV/hooks are IMPORTED from the #2094/#2333 suites, never
re-implemented (fmetrics.f_act, hooks.PositionEditHook, bank.norm_match,
issue2094_analysis.bootstrap_family_means_batched, decode_hooks
.generate_batch_ids).

Phases (each resumable; ``model_all`` = one model load for the model chain):

- ``bank``     stage kept/mined/segb rows from HF, sample N_Q_PER_CHAR
               questions per character, render each under story/chat/plain,
               tokenize, capture the ALL-task-layer context-end v_C bank, then
               run the in-process GATES (injection exactness incl. the
               hs-vs-block indexing seam, hidden-state padding parity, hooked
               HF-generate smoke). Gate FAIL exits RC_GATE (designed halt).
- ``anchors``  per context: K=10 unpatched temp-1.0 draws (+ per-draw v_A at
               the read layers) + one greedy 8-token opener (the #2333
               prefill-arm donor).
- ``grid``     the patched/prefill greedy cells (steered/within/null/prefill x
               variants), per-block rollout JSONL + v_A npz + ledger resume.
- ``screen``   F_act per cell + the pre-registered pair-clustered bootstrap
               screen (patch_common.screen_families) -> confirm selection.
- ``confirm``  temp-1.0 K=5 re-measure of the screened families (steered +
               matched null), labeled post-selection.
- ``upload``   HF upload (rollout text + tensors + reports) + git harvest of
               the eval JSONs + results sentinel.
- ``all``      parent mode: ensure model venv (issue2378_dispatch machinery),
               run ``model_all`` under the model interpreter, then ``upload``
               under the repo venv. Emits the single terminal ``[phase=done]``.

Layer-indexing seam (LOAD-BEARING): this task's stores index hidden states as
``hs[l]`` (hs[0]=embeddings), so task-layer L is the output of decoder BLOCK
L-1, while ``PositionEditHook(model, layer)`` hooks BLOCK ``layer``. Every
hook here therefore targets ``block = task_layer - 1``. VERIFICATION CAVEAT
(measured in the tiny e2e, repo transformers 4.57): ``output_hidden_states``
records each level BEFORE external forward hooks run, so a hooked forward's
``hs[task_layer]`` still shows the PRE-edit value even though the edit DOES
enter every downstream block (hs[task_layer+1] moves). The bank-phase
injection gate therefore verifies via a READER forward hook registered AFTER
the editor on the same block (later hooks receive the edited output): the
read value at (row, pos) must equal the donor at cos >= 0.999, norm ratio
[0.995, 1.005], be byte-unchanged at every other position, and
hs[task_layer+1][pos] must move vs the unhooked forward. Corollary: the
``all`` variant patches task layers 1..n-1 (blocks 0..n-2) — the recorded
``hs[n]`` is POST-final-norm (extraction.py caveat), a different space than
block n-1's raw output, so the last block is deliberately left unpatched.

Smoke blind-spot enumeration (smoke-blind-spots.md; the ``--tiny`` VM smoke):
- SUBSTITUTED implementation: ``--tiny`` builds a from-config qwen2-arch toy
  model (repo venv, transformers 4.57.6 has no qwen3_5) — the production
  ``Qwen3_5ForConditionalGeneration`` import + bf16 CUDA load run only under
  the pod model venv (covered pre-production by the bank-phase gates + the
  dispatch's ensure_model_venv env/engine smokes).
- DOWNGRADED gates: ``--tiny`` relaxes the 64-layer/5120-hidden config asserts
  (tiny shape) and skips the HBM preflight (CPU). The injection-exactness,
  padding-parity, and hooked-generate gates all RUN at tiny shape.
- PRODUCTION-ONLY paths: CUDA device routing; the model-venv interpreter
  resolution (``--phase all`` parent mode); production HF prefixes (the tiny
  e2e uploads under ``*_smoke`` prefixes via ``--hf-suffix``).

Pod-side contract: sentinel file + ``[phase=...]`` breadcrumbs only — NEVER a
``task.py`` shellout. The terminal ``[phase=done]`` is emitted ONCE by the
top-level invocation (child steps run with EPM_I2378_PATCH_CHILD=1 and their
stdout goes to per-step logs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch import (shared-VM thread caps + API keys)

import numpy as np  # noqa: E402

import issue2378_capture as cap  # noqa: E402
import issue2378_common as cm  # noqa: E402
import issue2378_gen as gen  # noqa: E402
import issue2378_patch_common as pc  # noqa: E402

if Path("/workspace").exists():  # pod clones; never rebinds a VM env
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

RC_GATE = 21  # designed gate halt (never a bare rc=1 — #1415 routing lesson)
GEN_BATCH_ROWS = 16
CAPTURE_BATCH_TOKENS = 16_384
CAPTURE_MAX_ROWS = 16
CHILD_ENV = "EPM_I2378_PATCH_CHILD"

GATE_COS_MIN = 0.999  # #2094 injection-exactness gate
GATE_NORM_LO, GATE_NORM_HI = 0.995, 1.005
PARITY_COS_MIN = 0.99  # fatal hidden-state padding-parity floor


def _log(msg: str) -> None:
    print(msg, flush=True)


def _phase_line(name: str) -> None:
    _log(f"[phase={name}]")


# ── config resolution ───────────────────────────────────────────────────────


def _n_layers(args) -> int:
    return int(args.tiny_layers) if args.tiny else cap.N_LAYERS


def _lstar(args) -> int:
    if args.lstar:
        return int(args.lstar)
    import issue2378_dispatch as D

    return D.resolve_lstar(Path(args.ledger_root))


def _read_layers(args) -> tuple[int, ...]:
    return pc.read_layers(_lstar(args), _n_layers(args))


def _out(args) -> Path:
    return Path(args.out_root)


def _hf_prefix(args, kind: str) -> str:
    base = pc.HF_STAGE_PREFIX if kind == "raw" else pc.HF_TENSOR_PREFIX
    return f"{base}{args.hf_suffix}"


# ── tiny-model support (VM smoke; see module docstring blind spots) ─────────


def _load_model_ctx(args) -> dict:
    """Production: issue2378_capture's fail-loud 27B loader (reused verbatim).
    ``--tiny``: from-config qwen2-arch toy on CPU (tiny-real CPU e2e)."""
    if not args.tiny:
        assert cm.MODEL_ID == "Qwen/Qwen3.6-27B", cm.MODEL_ID  # routing assert (#1738 r3)
        return cap._load_model_ctx(args)
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained(args.tiny_tokenizer)
    torch.manual_seed(cm.derived_seed("patch-tiny-model"))
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=int(args.tiny_hidden),
        intermediate_size=2 * int(args.tiny_hidden),
        num_hidden_layers=int(args.tiny_layers),
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    # bf16 like production — the reused capture kernel's uint16-bf16 encoding
    # asserts the dtype (cap._encode_bf16).
    model = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
    # Tiny-shape substitution (enumerated smoke blind spot): the reused
    # capture kernel allocates rows at cap.HIDDEN_SIZE — rebind it to the toy
    # width for this process. Production never enters this branch.
    cap.HIDDEN_SIZE = int(args.tiny_hidden)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return {
        "torch": torch,
        "model": model,
        "device": "cpu",
        "logits_kwargs": {},
        "pad_id": int(pad_id),
        "checked": True,  # tiny shape — skip the 64/5120 config assert
        "tok": tok,
    }


_MODEL_HOLDER: dict = {}


def _ensure_mctx(args) -> dict:
    """Process-wide memoized model context — ONE 27B load across the whole
    ``model_all`` chain (r17 Claude M1 / codex patch-model-all-reloads: four
    per-phase ``_load_model_ctx`` calls deserialized the checkpoint 4x and the
    driver-level HBM preflight could false-FAIL mid-chain on freed-but-
    allocator-retained prior-phase weights). Phases call this LAZILY, after
    computing their pending-unit set, so a fully-resumed phase never loads."""
    if "ctx" not in _MODEL_HOLDER:
        _MODEL_HOLDER["ctx"] = _load_model_ctx(args)
    return _MODEL_HOLDER["ctx"]


def _tok(args, mctx: dict | None = None):
    if mctx is not None and "tok" in mctx:
        return mctx["tok"]
    if args.tiny:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(args.tiny_tokenizer)
    return gen._get_tokenizer()


def _bank_digest(args) -> str:
    """sha256 over the bank files' on-disk bytes (bit-exact file inputs — safe
    to hash; regenerating the bank changes the digest and fail-louds every
    downstream StageLedger regime)."""
    import hashlib

    out = _out(args) / "bank"
    h = hashlib.sha256()
    for p in (out / "bank_rows.jsonl", out / "vc_bank.npz"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _openers_digest(openers: dict[str, list[int]]) -> str:
    """sha256 over the CONSUMED opener mapping (last-wins dict — stable under
    the benign duplicate re-append a mid-batch anchors resume can produce)."""
    import hashlib

    payload = json.dumps(sorted((k, list(v)) for k, v in openers.items()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── bank: rows + prompts + all-layer v_C capture + gates ────────────────────


def _sample_bank_questions(args) -> dict[str, list[dict]]:
    """Per character: N_Q_PER_CHAR seeded draws from kept ∩ mined ∩ segb rows.

    A kept storyq row supplies the in-story question (mined ``utterance``) and
    the story prompt pieces (``scene_pre_answer`` + segb ``opener_text``); the
    SAME utterance is rendered under chat + plain, so every pair shares its
    question exactly. All texts are model-GENERATED (SegA mining) — no
    real-corpus rows enter this round.
    """
    import random

    mined = gen._load_mined_rows(gen._rows_dir(args, "sega_mined", args.mined_dir))
    segb = gen._stage_kept_rows(gen._rows_dir(args, "segb"), None)
    kept_dir = Path(args.kept_dir)
    n_q = int(args.n_q_per_char)
    out: dict[str, list[dict]] = {}
    for cell in cm.STORY_Q_CELLS:
        char = cm.CELL_CHARACTER[cell]
        ids = gen._load_kept_ids(kept_dir, cell)
        rng = random.Random(cm.derived_seed("patch-bank", cell))
        rng.shuffle(ids)
        rows: list[dict] = []
        drops: Counter = Counter()
        for rid in ids:
            if len(rows) >= n_q:
                break
            m, r = mined.get(rid), segb.get(rid)
            if m is None or r is None:
                drops["mined_or_segb_missing"] += 1
                continue
            if m.get("opener_id") != r.get("opener_id"):
                drops["opener_id_mismatch"] += 1
                continue
            question = (m.get("utterance") or "").strip()
            if not question:
                drops["empty_utterance"] += 1
                continue
            rows.append(
                {
                    "qid": f"{cell}_{len(rows):02d}",
                    "char": char,
                    "source_row_id": rid,
                    "question": question,
                    "scene_pre_answer": m["scene_pre_answer"],
                    "opener_text": r["opener_text"],
                }
            )
        if len(rows) < n_q:
            raise RuntimeError(
                f"bank sampling: cell {cell} yielded {len(rows)} < {n_q} rows "
                f"(drops={dict(drops)}) — fail loud, never silently under-fill"
            )
        _log(f"[bank] {cell}: {len(rows)} questions sampled (drops={dict(drops)})")
        out[char] = rows
    return out


def _story_prompt(row: dict) -> str:
    """The story framing's generation prompt (SegB convention: scene + opener
    ending in an opening quote — the character's reply continues it)."""
    return row["scene_pre_answer"] + "\n\n" + row["opener_text"]


def _chat_prompt(tok, row: dict) -> str:
    return gen._render_chat(tok, row["question"])


def _plain_prompt(row: dict) -> str:
    return f"{cap.PLAIN_PREFIX}{row['question']}\n\nAssistant:"


def _bank_contexts(tok, qrows: dict[str, list[dict]]) -> list[dict]:
    ctxs: list[dict] = []
    for char in sorted(qrows):
        for row in qrows[char]:
            for framing, prompt in (
                ("story", _story_prompt(row)),
                ("chat", _chat_prompt(tok, row)),
                ("plain", _plain_prompt(row)),
            ):
                ids = tok(prompt, add_special_tokens=False)["input_ids"]
                assert len(ids) >= 2, (framing, row["qid"], len(ids))
                ctxs.append(
                    {
                        "ctx_id": pc.ctx_id(framing, row["qid"]),
                        # row_id aliases ctx_id: the reused _pack_batches
                        # sort key reads recs[i]["row_id"].
                        "row_id": pc.ctx_id(framing, row["qid"]),
                        "framing": framing,
                        "qid": row["qid"],
                        "char": char,
                        "question": row["question"],
                        "prompt_text": prompt,
                        "prompt_sha": cm.text_digest(prompt),
                        "input_ids": list(ids),
                        "n_tokens": len(ids),
                        "source_row_id": row["source_row_id"],
                    }
                )
    return ctxs


def _capture_vc_bank(args, mctx: dict, ctxs: list[dict]) -> dict[str, np.ndarray]:
    """ALL-task-layer context-end states per context: {ctx_id: (L, H) uint16}.

    Prompt-only forwards (causality makes the prompt-end state identical to
    the full-row state at that position); hs[l] task-layer indexing, matching
    the production capture convention (issue2378_capture._forward_chunk).
    Checkpointed per capture batch (r18: ~180 units > the T2 ~50-unit floor —
    a crash must not lose the whole bank): parts land under
    ``bank/vc_parts/partNNN.npz`` with a StageLedger keyed on the generating
    parameters; a resume re-captures only the missing parts.
    """
    import hashlib

    torch = mctx["torch"]
    model, dev = mctx["model"], mctx["device"]
    n_layers = _n_layers(args)
    parts_dir = _out(args) / "bank" / "vc_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ctxs_sha = hashlib.sha256(
        json.dumps([(c["ctx_id"], c["prompt_sha"]) for c in ctxs]).encode("utf-8")
    ).hexdigest()[:16]
    ledger = cm.StageLedger(
        parts_dir / "ledger.json",
        {
            "stage": "bank_capture",
            "model": "tiny" if args.tiny else cm.MODEL_ID,
            "n_layers": n_layers,
            "batch_tokens": int(args.batch_tokens),
            "max_batch_rows": int(args.max_batch_rows),
            "ctxs": ctxs_sha,
        },
    )
    # Token-budgeted packing (the production capture kernel's discipline):
    # output_hidden_states materializes (n_layers+1) x B x T x H — an
    # unbounded row-count batch of long rows OOMs (65 levels x 64k tokens x
    # 5120 bf16 ~ 43 GB); args.batch_tokens bounds B x T instead. The packing
    # is deterministic over the full ctx list, so part indices are
    # resume-stable.
    batches = cap._pack_batches(ctxs, args.batch_tokens, args.max_batch_rows)
    t0 = time.time()
    for bi, batch in enumerate(batches):
        key = f"part{bi:03d}"
        if ledger.is_done(key):
            continue
        t_max = max(ctxs[i]["n_tokens"] for i in batch)
        ids = torch.full((len(batch), t_max), mctx["pad_id"], dtype=torch.long)
        mask = torch.zeros((len(batch), t_max), dtype=torch.long)
        for j, i in enumerate(batch):
            ln = ctxs[i]["n_tokens"]
            ids[j, :ln] = torch.tensor(ctxs[i]["input_ids"], dtype=torch.long)
            mask[j, :ln] = 1
        with torch.no_grad():
            res = model(
                input_ids=ids.to(dev),
                attention_mask=mask.to(dev),
                output_hidden_states=True,
                **mctx["logits_kwargs"],
            )
        hs = res.hidden_states
        assert len(hs) == n_layers + 1, (len(hs), n_layers + 1)
        part: dict[str, np.ndarray] = {}
        for j, i in enumerate(batch):
            pos = ctxs[i]["n_tokens"] - 1
            vec = torch.stack([hs[layer][j, pos] for layer in range(1, n_layers + 1)])
            part[ctxs[i]["ctx_id"]] = cap._encode_bf16(torch, vec)
        del res, hs
        cap._atomic_savez(parts_dir / f"{key}.npz", **part)
        ledger.mark_done(key)
        cm.progress("bank.vc", bi + 1, len(batches), key, t0)
    out: dict[str, np.ndarray] = {}
    for p in sorted(parts_dir.glob("part*.npz")):
        z = np.load(p)
        out.update({k: z[k] for k in z.files})
    missing = {c["ctx_id"] for c in ctxs} - set(out)
    if missing:
        raise RuntimeError(f"bank capture parts incomplete: missing {sorted(missing)[:5]}")
    return out


def _vc(bank_vc: dict[str, np.ndarray], torch_mod, ctx: str, task_layer: int):
    """Decode one context's v_C at one task layer (fp32 torch, CPU)."""
    return cap.decode_bf16(bank_vc[ctx][task_layer - 1 : task_layer], torch_mod)[0].float()


def _vc_all(bank_vc: dict[str, np.ndarray], torch_mod, ctx: str):
    """(L_task, H) fp32 all-layer v_C for one context."""
    return cap.decode_bf16(bank_vc[ctx], torch_mod).float()


def _run_gates(args, mctx: dict, ctxs: list[dict], bank_vc: dict[str, np.ndarray]) -> dict:
    """In-process bank gates: injection exactness (hs-vs-block seam), padding
    parity (hidden-state fatal / greedy-token recorded), hooked-generate smoke."""
    from explore_persona_space.experiments.issue2094.hooks import PositionEditHook

    torch = mctx["torch"]
    model, dev = mctx["model"], mctx["device"]
    lstar = _lstar(args)
    by_id = {c["ctx_id"]: c for c in ctxs}
    report: dict = {"lstar": lstar, "read_layers": list(_read_layers(args)), "spots": []}

    # Spot pairs: first question of each character/framing rotation.
    ids_sorted = sorted(by_id)
    spots = []
    for k in range(min(6, len(ids_sorted) // 2)):
        rec, don = ids_sorted[2 * k], ids_sorted[2 * k + 1]
        if rec != don:
            spots.append((rec, don))
    assert spots, "no gate spots derivable from the bank"

    class _Reader:
        """Reads the block's (post-editor) output — registered AFTER the
        editor, so it receives the edited tensor (pytorch hook ordering)."""

        def __init__(self):
            self.out = None

        def __call__(self, _mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            self.out = t.detach().float().cpu()

    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    blocks, _, _ = _resolve_decoder_blocks(model)
    block = blocks[lstar - 1]  # block = task_layer - 1 (the module-docstring seam)
    downstream_level = lstar + 1
    assert downstream_level <= _n_layers(args), (lstar, _n_layers(args))

    ok_all = True
    for rec_id, don_id in spots:
        rec = by_id[rec_id]
        donor_vec = _vc(bank_vc, torch, don_id, lstar).to(dev)
        pos = rec["n_tokens"] - 1
        ids_t = torch.tensor([rec["input_ids"]], dtype=torch.long, device=dev)
        reader = _Reader()
        rh = block.register_forward_hook(reader)
        try:
            with torch.no_grad():
                base = model(input_ids=ids_t, output_hidden_states=True, **mctx["logits_kwargs"])
        finally:
            rh.remove()
        base_out = reader.out[0]
        hook = PositionEditHook(model, lstar - 1)
        hook.install()
        reader = _Reader()
        rh = block.register_forward_hook(reader)  # AFTER the editor -> sees the edit
        try:
            hook.arm_batch([rec["n_tokens"]], [[pos]], [donor_vec.unsqueeze(0)], mode="replace")
            hook.arm(rec["n_tokens"])
            with torch.no_grad():
                patched = model(input_ids=ids_t, output_hidden_states=True, **mctx["logits_kwargs"])
        finally:
            rh.remove()
            hook.remove()
        hooked_out = reader.out[0]
        got = hooked_out[pos]
        cos = float(
            torch.nn.functional.cosine_similarity(got, donor_vec.float().cpu().to(got.device), 0)
        )
        ratio = float(got.norm() / donor_vec.float().norm().clamp_min(1e-30))
        others = torch.ones(hooked_out.shape[0], dtype=torch.bool)
        others[pos] = False
        elsewhere = float((hooked_out[others] - base_out[others]).abs().max())
        downstream_moved = float(
            (
                patched.hidden_states[downstream_level][0, pos].float()
                - base.hidden_states[downstream_level][0, pos].float()
            )
            .abs()
            .max()
        )
        spot_ok = (
            cos >= GATE_COS_MIN
            and GATE_NORM_LO <= ratio <= GATE_NORM_HI
            and elsewhere == 0.0
            and downstream_moved > 0.0
        )
        ok_all = ok_all and spot_ok
        report["spots"].append(
            {
                "recipient": rec_id,
                "donor": don_id,
                "cos": cos,
                "norm_ratio": ratio,
                "max_abs_elsewhere": elsewhere,
                "downstream_moved_max_abs": downstream_moved,
                "n_edits": hook.n_edits,
                "ok": spot_ok,
            }
        )
        del base, patched

    # Padding parity: shortest + longest context, unhooked.
    from explore_persona_space.experiments.issue2333.decode_hooks import generate_batch_ids

    short = min(ctxs, key=lambda c: c["n_tokens"])
    longc = max(ctxs, key=lambda c: c["n_tokens"])
    with torch.no_grad():
        solo = model(
            input_ids=torch.tensor([short["input_ids"]], dtype=torch.long, device=dev),
            output_hidden_states=True,
            **mctx["logits_kwargs"],
        )
    t_max = longc["n_tokens"]
    ids2 = torch.full((2, t_max), mctx["pad_id"], dtype=torch.long)
    mask2 = torch.zeros((2, t_max), dtype=torch.long)
    for j, c in enumerate((short, longc)):
        ids2[j, t_max - c["n_tokens"] :] = torch.tensor(c["input_ids"], dtype=torch.long)
        mask2[j, t_max - c["n_tokens"] :] = 1
    with torch.no_grad():
        duo = model(
            input_ids=ids2.to(dev),
            attention_mask=mask2.to(dev),
            output_hidden_states=True,
            **mctx["logits_kwargs"],
        )
    v_solo = solo.hidden_states[lstar][0, short["n_tokens"] - 1].float()
    v_duo = duo.hidden_states[lstar][0, t_max - 1].float()
    par_cos = float(torch.nn.functional.cosine_similarity(v_solo, v_duo, 0))
    report["padding_parity_cos"] = par_cos
    report["padding_parity_ok"] = bool(par_cos >= PARITY_COS_MIN)
    ok_all = ok_all and report["padding_parity_ok"]
    del solo, duo

    # Greedy-token parity (RECORDED, non-fatal — bf16 left-pad batch geometry
    # legitimately jitters logits; feedback_gate_reads_batch_geometry).
    tok = _tok(args, mctx)
    g_solo = generate_batch_ids(model, tok, [short["input_ids"]], greedy=True, max_new_tokens=8)
    g_duo = generate_batch_ids(
        model, tok, [short["input_ids"], longc["input_ids"]], greedy=True, max_new_tokens=8
    )
    report["greedy_parity_match"] = bool(g_solo[0][0]["gen_ids"] == g_duo[0][0]["gen_ids"])

    # Hooked-generate smoke: replace hook armed through a real generate call.
    rec = ctxs[0]
    donor_vec = _vc(bank_vc, torch, ctxs[1]["ctx_id"], lstar).to(dev)
    hook = PositionEditHook(model, lstar - 1)
    hook.install()
    try:
        hook.arm_batch(
            [rec["n_tokens"]], [[rec["n_tokens"] - 1]], [donor_vec.unsqueeze(0)], mode="replace"
        )
        hook.arm(rec["n_tokens"])
        g = generate_batch_ids(model, tok, [rec["input_ids"]], greedy=True, max_new_tokens=8)
    finally:
        hook.remove()
    report["hooked_generate_n_edits"] = hook.n_edits
    report["hooked_generate_n_tokens"] = len(g[0][0]["gen_ids"])
    hooked_ok = hook.n_edits >= 1 and len(g[0][0]["gen_ids"]) >= 1
    ok_all = ok_all and hooked_ok
    report["hooked_generate_ok"] = bool(hooked_ok)
    report["ok"] = bool(ok_all)
    return report


def phase_bank(args) -> int:
    _phase_line("patch_bank")
    out = _out(args) / "bank"
    out.mkdir(parents=True, exist_ok=True)
    tokz = _tok(args)
    qrows = _sample_bank_questions(args)
    ctxs = _bank_contexts(tokz, qrows)
    mctx = _ensure_mctx(args)  # bank always needs the model: gates re-verify on every entry
    bank_vc = _capture_vc_bank(args, mctx, ctxs)
    np.savez(out / "vc_bank.npz", **bank_vc)
    rows_path = out / "bank_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for c in ctxs:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    gate = _run_gates(args, mctx, ctxs, bank_vc)
    gate["metadata"] = cm.run_metadata({"phase": "patch_bank", "n_contexts": len(ctxs)})
    cm.atomic_write_json(out / "gate_report.json", gate)
    _log(f"[bank] contexts={len(ctxs)} gate_ok={gate['ok']}")
    if not gate["ok"]:
        _log("[bank] GATE FAIL — designed halt (see gate_report.json)")
        return RC_GATE
    return 0


def _load_bank(args) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Bank rows + v_C bank with exact runtime KEY-COVERAGE asserts (r17 codex
    patch-cache-key-coverage): unique row ctx ids, and vc_bank.npz keys
    EXACTLY equal to the row ctx-id set — a partial/stale/mismatched prior
    bank fails loud here, BEFORE any 27B model load."""
    out = _out(args) / "bank"
    rows = list(cm.iter_jsonl(out / "bank_rows.jsonl"))
    if not rows:
        raise RuntimeError("empty bank_rows.jsonl (fail loud)")
    ids = [r["ctx_id"] for r in rows]
    dupes = sorted(k for k, v in Counter(ids).items() if v > 1)
    if dupes:
        raise RuntimeError(f"duplicate bank ctx ids (stale/partial bank): {dupes[:5]}")
    z = np.load(out / "vc_bank.npz")
    vc = {k: z[k] for k in z.files}
    missing = sorted(set(ids) - set(vc))
    extra = sorted(set(vc) - set(ids))
    if missing or extra:
        raise RuntimeError(
            "vc_bank.npz key-coverage mismatch vs bank_rows.jsonl: "
            f"missing={missing[:5]} extra={extra[:5]} (n_rows={len(ids)}, n_vc={len(vc)}) "
            "— stale/partial bank; re-run --phase bank into a fresh out-root"
        )
    return rows, vc


# ── shared rollout post-processing ──────────────────────────────────────────


def _extract_answer(framing: str, text: str) -> tuple[str | None, str | None]:
    """(answer, drop_reason) per target-framing stop convention (parent recipe)."""
    if framing == "story":
        close = gen._mine_closing_quote(text)
        if close is None:
            return None, "cap_hit_no_close"
        ans = text[:close].strip()
        return (ans, None) if ans else (None, "empty_answer")
    if framing == "plain":
        for stop in cm.PLAIN_STOP:
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx]
        ans = text.strip()
        return (ans, None) if ans else (None, "empty_answer")
    ans = text.strip()
    return (ans, None) if ans else (None, "empty_answer")


def _va_capture(args, mctx, tokz, items: list[dict]) -> np.ndarray | None:
    """Span-mean answer states at the read layers for (prompt, answer) items.

    Reuses the production capture path verbatim (_tokenize_and_positions +
    _forward_chunk on assembled final_texts — the store's v_A convention).
    Returns (n, n_read_layers, H) uint16 rows aligned with ``items``; a row
    whose spans do not resolve gets NaN-encoded zeros + item["va_dropped"].
    """
    layers = list(_read_layers(args))
    rows = []
    for k, it in enumerate(items):
        prompt, ans = it["prompt_text"], it["answer"]
        joiner = " " if it["framing"] == "plain" else ""
        final = prompt + joiner + ans
        rows.append(
            {
                "row_id": f"r{k}",
                "final_text": final,
                "answer_lo_char": len(prompt) + len(joiner),
                "answer_hi_char": len(final),
                # v_P is unused this round; anchor it at the answer start so
                # the reused kernel resolves it (char 0 can never resolve).
                "prefix_char": len(prompt) + len(joiner),
                "_item": k,
            }
        )
    kept, drops = cap._tokenize_and_positions(tokz, rows, max_tokens=int(args.capture_max_tokens))
    if drops:
        _log(f"[va] span drops: {dict(drops)}")
    return _va_from_recs(args, mctx, items, kept, layers)


def _va_from_recs(args, mctx, items: list[dict], kept: list[dict], layers: list[int]):
    """Shared v_A assembly: forward the kept recs, align rows back to items."""
    for it in items:
        it["va_dropped"] = True
    if not kept:
        return None
    per_layer = cap._forward_chunk(args, mctx, kept, layers)
    hdim = per_layer[layers[0]]["v_A"].shape[-1]
    out = np.zeros((len(items), len(layers), hdim), dtype=np.uint16)
    for j, r in enumerate(kept):
        items[r["_item"]]["va_dropped"] = False
        for li, layer in enumerate(layers):
            out[r["_item"], li] = per_layer[layer]["v_A"][j]
    return out


def _prefill_capture_rec(k: int, prompt_ids: list[int], completion_ids: list[int]) -> dict:
    """One exact-token-ID capture rec for a prefill row (#2333 convention):
    input ids = prompt + completion VERBATIM (never re-tokenized text), v_A
    span = the whole kept completion — matching issue2333_run's
    ``span = (len(base), len(full))`` — v_C at the last prompt position."""
    n_p, n = len(prompt_ids), len(prompt_ids) + len(completion_ids)
    assert completion_ids, "empty completion ids"
    return {
        "row_id": f"r{k}",
        "input_ids": [*prompt_ids, *completion_ids],
        "n_tokens": n,
        "v_C_pos": n_p - 1,
        "v_P_pos": n_p,  # v_P unused this round; anchored at the answer start
        "ans_lo": n_p,
        "ans_hi": n,
        "_item": k,
    }


def _va_capture_ids(args, mctx, items: list[dict]) -> np.ndarray | None:
    """Exact-token-ID v_A capture for prefill rows (r17 codex
    patch-prefill-token-identity-loss): consumes each item's
    ``_capture_ids`` = {"prompt": ids, "completion": ids} carried from the
    generation call — no decode -> re-tokenize round trip anywhere. Reuses
    the production forward kernel (cap._forward_chunk) verbatim."""
    layers = list(_read_layers(args))
    max_tokens = int(args.capture_max_tokens)
    recs: list[dict] = []
    drops: Counter = Counter()
    for k, it in enumerate(items):
        pid = it["_capture_ids"]["prompt"]
        cid = it["_capture_ids"]["completion"]
        if not cid:
            drops["empty_completion"] += 1
            continue
        if len(pid) + len(cid) > max_tokens:
            drops["over_length"] += 1
            continue
        recs.append(_prefill_capture_rec(k, pid, cid))
    if drops:
        _log(f"[va-ids] drops: {dict(drops)}")
    return _va_from_recs(args, mctx, items, recs, layers)


# ── anchors ─────────────────────────────────────────────────────────────────


def _anchor_units(ctxs: list[dict]) -> list[tuple[str, list[dict]]]:
    """Batch-STABLE anchor units: fixed per-framing chunks over the FULL bank
    order (never a todo-filtered list), so a resume re-runs identical batches
    with identical per-unit seeds (r17 codex: a todo-derived batch shifted the
    remaining contexts' draws on resume), and every batch is single-framing
    (exact per-framing cap + stop set — never a mixed-cap max)."""
    by_framing: dict[str, list[dict]] = {}
    for c in ctxs:
        by_framing.setdefault(c["framing"], []).append(c)
    units: list[tuple[str, list[dict]]] = []
    for framing in sorted(by_framing):
        cs = by_framing[framing]  # bank order is deterministic
        for k in range(0, len(cs), GEN_BATCH_ROWS):
            units.append((f"{framing}|b{k // GEN_BATCH_ROWS:03d}", cs[k : k + GEN_BATCH_ROWS]))
    return units


def phase_anchors(args) -> int:
    _phase_line("patch_anchors")
    from explore_persona_space.experiments.issue2333.decode_hooks import generate_batch_ids

    out = _out(args) / "anchors"
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    (out / "va").mkdir(parents=True, exist_ok=True)
    ctxs, _ = _load_bank(args)
    draws = int(args.anchor_draws)
    regime = {
        "stage": "anchors",
        "draws": draws,
        "temperature": cm.TEMPERATURE,
        "opener_tokens": pc.OPENER_TOKENS,
        "read_layers": list(_read_layers(args)),
        "n_q_per_char": int(args.n_q_per_char),
        "capture_max_tokens": int(args.capture_max_tokens),
        "bank": _bank_digest(args),
        "tiny": bool(args.tiny),
    }
    ledger = cm.StageLedger(out / "ledger.json", regime)
    units = _anchor_units(ctxs)
    todo = [(key, batch) for key, batch in units if not ledger.is_done(key)]
    _log(f"[anchors] {len(todo)}/{len(units)} units to run (resume skips the rest)")
    if not todo:
        return 0
    mctx = _ensure_mctx(args)  # lazy: a fully-resumed phase never loads
    tokz = _tok(args, mctx)
    t0 = time.time()
    openers_path = out / "openers.jsonl"
    for u, (unit_key, batch) in enumerate(todo):
        framing = batch[0]["framing"]
        ids = [c["input_ids"] for c in batch]
        cap_tokens = 16 if args.tiny else pc.FRAMING_MAX_TOKENS[framing]
        per_draw = generate_batch_ids(
            model=mctx["model"],
            tokenizer=tokz,
            rows_ids=ids,
            n=draws,
            max_new_tokens=cap_tokens,
            temperature=cm.TEMPERATURE,
            seed_base=cm.derived_seed("patch-anchors", unit_key),
            stop_strings=pc.stop_strings_for(framing),
        )
        # Openers deliberately carry NO stop set: the #2333 donor scheme takes
        # the first PREFILL_K greedy tokens verbatim, stops included.
        openers = generate_batch_ids(
            model=mctx["model"],
            tokenizer=tokz,
            rows_ids=ids,
            n=1,
            greedy=True,
            max_new_tokens=pc.OPENER_TOKENS,
        )
        for b, c in enumerate(batch):
            items = []
            for d in range(draws):
                row = per_draw[d][b]
                ans, drop = _extract_answer(c["framing"], row["text"])
                items.append(
                    {
                        "ctx_id": c["ctx_id"],
                        "framing": c["framing"],
                        "draw": d,
                        "prompt_text": c["prompt_text"],
                        "answer": ans or "",
                        "gen_text": row["text"],
                        "n_completion_tokens": row["n_completion_tokens"],
                        "hit_eos": row["hit_eos"],
                        "hit_stop": pc.hit_stop(c["framing"], row["text"]),
                        "drop_reason": drop,
                    }
                )
            live = [it for it in items if it["drop_reason"] is None]
            va = _va_capture(args, mctx, tokz, live) if live else None
            slug = c["ctx_id"].replace(":", "_")
            with (out / "rollouts" / f"{slug}.jsonl").open("w", encoding="utf-8") as fh:
                for it in items:
                    fh.write(json.dumps(it, ensure_ascii=False) + "\n")
            if va is not None:
                kept_draws = [it["draw"] for it in live if not it["va_dropped"]]
                keep_rows = [j for j, it in enumerate(live) if not it["va_dropped"]]
                cap._atomic_savez(
                    out / "va" / f"{slug}.npz",
                    va=va[keep_rows],
                    draws=np.asarray(kept_draws, dtype=np.int32),
                )
            with openers_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ctx_id": c["ctx_id"],
                            "opener_ids": openers[0][b]["gen_ids"][: pc.PREFILL_K],
                            "opener_text": openers[0][b]["text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        ledger.mark_done(unit_key)
        cm.progress("anchors", u + 1, len(todo), unit_key, t0)
    return 0


def _anchor_va(args) -> dict[str, np.ndarray]:
    """{ctx_id: (K_kept, n_read_layers, H) uint16} anchor answer states."""
    out = _out(args) / "anchors" / "va"
    got: dict[str, np.ndarray] = {}
    for p in sorted(out.glob("*.npz")):
        z = np.load(p)
        framing, qid = p.stem.split("_", 1)
        got[f"{framing}:{qid}"] = z["va"]
    if not got:
        raise RuntimeError(f"no anchor v_A under {out} (empty selection — fail loud)")
    return got


def _openers(args, expected_ctx_ids=None) -> dict[str, list[int]]:
    """Opener-token map, with exact key coverage against the bank ctx-id set
    when ``expected_ctx_ids`` is given (r17 codex patch-cache-key-coverage —
    checked BEFORE any model load; last-wins over benign duplicate rows)."""
    path = _out(args) / "anchors" / "openers.jsonl"
    got: dict[str, list[int]] = {}
    for row in cm.iter_jsonl(path):
        got[row["ctx_id"]] = list(row["opener_ids"])
    if not got:
        raise RuntimeError(f"no openers at {path} (fail loud)")
    if expected_ctx_ids is not None:
        missing = sorted(set(expected_ctx_ids) - set(got))
        extra = sorted(set(got) - set(expected_ctx_ids))
        if missing or extra:
            raise RuntimeError(
                f"openers.jsonl key-coverage mismatch vs the bank: missing={missing[:5]} "
                f"extra={extra[:5]} — stale/partial anchors; re-run --phase anchors"
            )
    return got


# ── grid (patched + prefill greedy cells) ───────────────────────────────────


def _donor_payload(args, torch_mod, bank_vc, cell: dict, dmaps: dict) -> tuple[str, object]:
    """(donor_ctx_id, payload) for one hooked cell.

    steered: the SOURCE context's own v_C (real state, alpha=1 replace).
    within:  the TARGET framing's wrong-question state (same char grain).
    null:    the SOURCE framing's wrong-question state, norm-matched to the
             recipient's own v_C (the #2094 replace-null realization).
    Payload: (H,) fp32 for variant=lstar; (L_task, H) fp32 for variant=all.
    """
    from explore_persona_space.experiments.issue2094.bank import norm_match

    lstar = _lstar(args)
    variant, arm = cell["variant"], cell["arm"]
    src_framing = cell["src"].split(":", 1)[0]
    tgt_framing = cell["tgt"].split(":", 1)[0]
    if arm == "steered":
        donor_ctx = cell["src"]
    elif arm == "within":
        donor_ctx = pc.ctx_id(tgt_framing, dmaps[(tgt_framing, cell["char"])][cell["qid"]])
    elif arm == "null":
        donor_ctx = pc.ctx_id(src_framing, dmaps[(src_framing, cell["char"])][cell["qid"]])
    else:
        raise AssertionError(f"no donor payload for arm {arm}")
    if variant == "lstar":
        vec = _vc(bank_vc, torch_mod, donor_ctx, lstar)
        if arm == "null":
            rec = _vc(bank_vc, torch_mod, cell["tgt"], lstar)
            vec = norm_match(vec.unsqueeze(0), rec.unsqueeze(0))[0]
        return donor_ctx, vec
    mat = _vc_all(bank_vc, torch_mod, donor_ctx)
    if arm == "null":
        rec = _vc_all(bank_vc, torch_mod, cell["tgt"])
        mat = norm_match(mat, rec)  # per-task-layer norm match (rows = layers)
    return donor_ctx, mat


def _run_hooked_block(
    args, mctx, tokz, block_cells, by_ctx, bank_vc, dmaps, greedy, draws, seed_tag
):
    """One generation block sharing (variant, tgt framing): returns rollout rows.

    Hooks: block index = task_layer - 1 (the module-docstring seam). One
    generate call per draw (the prefill latch resets on re-arm)."""
    from explore_persona_space.experiments.issue2094.hooks import (
        PositionEditHook,
        PositionEditHookStack,
    )

    torch = mctx["torch"]
    model, dev = mctx["model"], mctx["device"]
    lstar = _lstar(args)
    n_layers = _n_layers(args)
    variant = block_cells[0]["variant"]
    tgt_framing = block_cells[0]["tgt"].split(":", 1)[0]
    assert all(c["variant"] == variant for c in block_cells)
    rows_ids, row_lengths, positions, donors = [], [], [], []
    for c in block_cells:
        rec = by_ctx[c["tgt"]]
        rows_ids.append(rec["input_ids"])
        row_lengths.append(rec["n_tokens"])
        positions.append([rec["n_tokens"] - 1])
        donor_ctx, payload = _donor_payload(args, torch, bank_vc, c, dmaps)
        c["donor_ctx"] = donor_ctx
        donors.append(payload)
    cap_tokens = 16 if args.tiny else pc.FRAMING_MAX_TOKENS[tgt_framing]
    if variant == "lstar":
        hook = PositionEditHook(model, lstar - 1)
        hook.install()
        hook.arm_batch(
            row_lengths,
            positions,
            [d.unsqueeze(0).to(dev) for d in donors],
            mode="replace",
        )
    else:
        # Task layers 1..n-1 (blocks 0..n-2): the recorded hs[n] is POST-final-
        # norm — a different space than block n-1's raw output — so the last
        # block stays unpatched (module-docstring seam corollary).
        hook = PositionEditHookStack(
            [PositionEditHook(model, layer - 1) for layer in range(1, n_layers)]
        )
        hook.install()
        hook.arm_batch_per_layer(
            row_lengths,
            positions,
            [[d[layer - 1].unsqueeze(0).to(dev) for d in donors] for layer in range(1, n_layers)],
            mode="replace",
        )
    from explore_persona_space.experiments.issue2333.decode_hooks import generate_batch_ids

    out_rows = []
    try:
        t_pad = max(row_lengths)
        for d in range(draws):
            hook.arm(t_pad)
            g = generate_batch_ids(
                model=model,
                tokenizer=tokz,
                rows_ids=rows_ids,
                n=1,
                greedy=greedy,
                temperature=cm.TEMPERATURE,
                max_new_tokens=cap_tokens,
                seed_base=cm.derived_seed(seed_tag, block_cells[0]["cell_id"], d),
                stop_strings=pc.stop_strings_for(tgt_framing),
            )
            for b, c in enumerate(block_cells):
                row = g[0][b]
                ans, drop = _extract_answer(tgt_framing, row["text"])
                out_rows.append(
                    {
                        **{
                            k: c[k]
                            for k in (
                                "cell_id",
                                "arm",
                                "variant",
                                "src",
                                "tgt",
                                "qid",
                                "char",
                                "pair_type",
                                "direction",
                                "family",
                                "donor_ctx",
                            )
                        },
                        "draw": d,
                        "framing": tgt_framing,
                        "prompt_text": by_ctx[c["tgt"]]["prompt_text"],
                        "answer": ans or "",
                        "gen_text": row["text"],
                        "n_completion_tokens": row["n_completion_tokens"],
                        "hit_eos": row["hit_eos"],
                        "hit_stop": pc.hit_stop(tgt_framing, row["text"]),
                        "drop_reason": drop,
                        "prefill_k": None,
                    }
                )
    finally:
        hook.remove()
    return out_rows


_ROW_CELL_KEYS = (
    "cell_id",
    "arm",
    "variant",
    "src",
    "tgt",
    "qid",
    "char",
    "pair_type",
    "direction",
    "family",
    "donor_ctx",
)
OPENER_EMPTY_DROP = "opener_empty"


def _check_opener_drop_floor(n_dropped: int, n_prefill_cells: int) -> None:
    """Fail-loud floor over counted opener_empty drops (r19): an isolated
    all-stop-text greedy opener is a recorded data pathology, but above
    max(1, 5% of the prefill cells this invocation processes) the stop
    wiring itself is suspect and the run must still crash."""
    floor = max(1.0, 0.05 * n_prefill_cells)
    if n_dropped > floor:
        raise AssertionError(
            f"{n_dropped} opener_empty drops exceed floor {floor:g} "
            f"(max(1, 5% of {n_prefill_cells} prefill cells)) — systematic "
            "stop-wiring pathology (fail loud)"
        )


def _prefill_drop_row(c, tgt_framing, prompt_text, d) -> dict:
    """Counted-drop row for a prefill cell whose SOURCE opener is EMPTY (the
    greedy reply was entirely stop text under the v18 stop wiring, r19): the
    cell gets no generation slot and no v_A; the row lands in the rollout
    JSONL and the screen report's ``dropped`` bucket (same counted-bucket
    pattern as ``families_no_null_pairs``), never a crash."""
    return {
        **{k: c[k] for k in _ROW_CELL_KEYS},
        "draw": d,
        "framing": tgt_framing,
        "prompt_text": prompt_text,
        "answer": "",
        "gen_text": "",
        "n_completion_tokens": 0,
        "hit_eos": False,
        "hit_stop": False,
        "drop_reason": OPENER_EMPTY_DROP,
        "prefill_k": pc.PREFILL_K,
        "va_span": "completion_ids",
    }


def _prefill_row(tokz, c, tgt_framing, prompt_ids, opener_ids, gen_row, d) -> dict:
    """One prefill rollout row (#2333 token-identity convention, r17 codex
    patch-prefill-token-identity-loss): the judged reply is the ONE-SHOT
    decode of ``[opener_ids + gen_ids]`` — a split decode can corrupt a
    multi-byte / cleanup-space seam between the segments (issue2333_run r1
    blocker) — and the EXACT combined ids ride ``_capture_ids`` for the v_A
    capture, never a re-tokenized text round trip."""
    completion_ids = [*opener_ids, *gen_row["gen_ids"]]
    text = tokz.decode(completion_ids, skip_special_tokens=True)
    ans, drop = _extract_answer(tgt_framing, text)
    return {
        **{k: c[k] for k in _ROW_CELL_KEYS},
        "draw": d,
        "framing": tgt_framing,
        "prompt_text": None,  # filled by the caller (by_ctx lookup)
        "answer": ans or "",
        "gen_text": text,
        "n_completion_tokens": len(completion_ids),
        "hit_eos": gen_row["hit_eos"],
        "hit_stop": pc.hit_stop(tgt_framing, text),
        "drop_reason": drop,
        "prefill_k": pc.PREFILL_K,
        # v_A span convention for this arm: the WHOLE kept completion in ID
        # space (issue2333_run span=(len(base), len(full))), never the
        # re-tokenized trimmed text; stop_strings bound post-stop residue.
        "va_span": "completion_ids",
        "_capture_ids": {"prompt": list(prompt_ids), "completion": completion_ids},
    }


def _run_prefill_block(args, mctx, tokz, block_cells, by_ctx, openers, greedy, draws, seed_tag):
    """Prefill arm: TARGET prompt + the SOURCE's first-k greedy opener TOKEN
    ids (token-id concatenation, #2333 convention); no hook. generate_batch_ids
    asserts the padded row tail carries [prompt+opener] ids verbatim (the
    #2333 exact-ID assert)."""
    from explore_persona_space.experiments.issue2333.decode_hooks import generate_batch_ids

    tgt_framing = block_cells[0]["tgt"].split(":", 1)[0]
    rows_ids, opener_by_b, kept_cells, empty_cells = [], [], [], []
    for c in block_cells:
        c["donor_ctx"] = c["src"]
        opener = list(openers[c["src"]][: pc.PREFILL_K])
        if not opener:
            # r19 counted drop: an all-stop-text greedy reply yields an empty
            # opener (isolated data pathology) — skip the cell, record it;
            # phase_grid enforces the systematic-pathology floor.
            empty_cells.append(c)
            continue
        rows_ids.append(list(by_ctx[c["tgt"]]["input_ids"]) + opener)
        opener_by_b.append(opener)
        kept_cells.append(c)
    cap_tokens = 16 if args.tiny else pc.FRAMING_MAX_TOKENS[tgt_framing]
    out_rows = []
    for d in range(draws):
        for c in empty_cells:
            out_rows.append(_prefill_drop_row(c, tgt_framing, by_ctx[c["tgt"]]["prompt_text"], d))
    if not kept_cells:
        return out_rows  # zero-dispatch guard: never generate on an empty batch
    for d in range(draws):
        g = generate_batch_ids(
            model=mctx["model"],
            tokenizer=tokz,
            rows_ids=rows_ids,
            n=1,
            greedy=greedy,
            temperature=cm.TEMPERATURE,
            max_new_tokens=cap_tokens,
            seed_base=cm.derived_seed(seed_tag, block_cells[0]["cell_id"], d),
            stop_strings=pc.stop_strings_for(tgt_framing),
        )
        for b, c in enumerate(kept_cells):
            row = _prefill_row(
                tokz, c, tgt_framing, by_ctx[c["tgt"]]["input_ids"], opener_by_b[b], g[0][b], d
            )
            row["prompt_text"] = by_ctx[c["tgt"]]["prompt_text"]
            out_rows.append(row)
    return out_rows


def _derangement_maps(ctxs: list[dict]) -> dict:
    """{(framing, char): {qid: donor_qid}} seeded fixed-point-free maps."""
    groups: dict[tuple[str, str], list[str]] = {}
    for c in ctxs:
        groups.setdefault((c["framing"], c["char"]), []).append(c["qid"])
    return {key: pc.derangement(sorted(set(qids)), key) for key, qids in groups.items()}


def _grid_cells(args, ctxs: list[dict]) -> list[dict]:
    qids_by_char: dict[str, list[str]] = {}
    for c in ctxs:
        if c["framing"] == "story":
            qids_by_char.setdefault(c["char"], []).append(c["qid"])
    cells = pc.enumerate_cells({k: sorted(v) for k, v in qids_by_char.items()})
    if args.cells_limit:
        # smoke subset: >= 1 cell per (arm, variant) class (smoke coverage law)
        per_class: dict[tuple[str, str], list[dict]] = {}
        for c in cells:
            per_class.setdefault((c["arm"], c["variant"]), []).append(c)
        subset = []
        for key in sorted(per_class):
            subset.extend(per_class[key][: int(args.cells_limit)])
        cells = subset
    return cells


def _write_unit_outputs(out: Path, unit_key: str, rows: list[dict], live: list[dict], va) -> None:
    """Rollout JSONL (private ``_capture_ids`` stripped) + kept-row v_A npz."""
    slug = unit_key.replace("|", "_")
    with (out / "rollouts" / f"{slug}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                json.dumps({k: v for k, v in r.items() if k != "_capture_ids"}, ensure_ascii=False)
                + "\n"
            )
    if va is not None:
        keep = [j for j, it in enumerate(live) if not it["va_dropped"]]
        cap._atomic_savez(
            out / "va" / f"{slug}.npz",
            va=va[keep],
            cell_ids=np.asarray([f"{live[j]['cell_id']}#{live[j]['draw']}" for j in keep]),
        )


def phase_grid(args) -> int:
    _phase_line("patch_grid")
    out = _out(args) / "grid"
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    (out / "va").mkdir(parents=True, exist_ok=True)
    ctxs, bank_vc = _load_bank(args)
    by_ctx = {c["ctx_id"]: c for c in ctxs}
    # Key-coverage checks run BEFORE any model load (patch-cache-key-coverage).
    openers = _openers(args, expected_ctx_ids=[c["ctx_id"] for c in ctxs])
    dmaps = _derangement_maps(ctxs)
    cells = _grid_cells(args, ctxs)
    regime = {
        "stage": "grid",
        "lstar": _lstar(args),
        "read_layers": list(_read_layers(args)),
        "prefill_k": pc.PREFILL_K,
        "n_q_per_char": int(args.n_q_per_char),
        "cells_limit": int(args.cells_limit or 0),
        "capture_max_tokens": int(args.capture_max_tokens),
        "bank": _bank_digest(args),
        "openers": _openers_digest(openers),
        "tiny": bool(args.tiny),
    }
    ledger = cm.StageLedger(out / "ledger.json", regime)

    blocks: dict[str, list[dict]] = {}
    for c in cells:
        hookkey = c["variant"] if c["arm"] != "prefill" else "prefill"
        blocks.setdefault(f"{hookkey}|{c['tgt'].split(':', 1)[0]}", []).append(c)
    units: list[tuple[str, list[dict]]] = []
    for key in sorted(blocks):
        cs = sorted(blocks[key], key=lambda c: c["cell_id"])
        for k in range(0, len(cs), GEN_BATCH_ROWS):
            units.append((f"{key}|{k // GEN_BATCH_ROWS:03d}", cs[k : k + GEN_BATCH_ROWS]))
    todo = [(key, cs) for key, cs in units if not ledger.is_done(key)]
    _log(f"[grid] {len(todo)}/{len(units)} units to run (resume skips the rest)")
    if not todo:
        return 0
    # r19 counted-drop accounting (OUTPUT only — no regime key, so resumed runs
    # skip completed units unchanged): floor over the prefill cells THIS
    # invocation processes; resume-skipped units keep their committed rollouts.
    n_prefill_cells = sum(len(cs) for _, cs in todo if cs[0]["arm"] == "prefill")
    dropped_opener_empty: list[str] = []
    mctx = _ensure_mctx(args)  # lazy: a fully-resumed phase never loads
    tokz = _tok(args, mctx)
    t0 = time.time()
    for u, (unit_key, block_cells) in enumerate(todo):
        prefill = block_cells[0]["arm"] == "prefill"
        if prefill:
            rows = _run_prefill_block(
                args, mctx, tokz, block_cells, by_ctx, openers, True, 1, "patch-grid"
            )
            unit_empty = sorted(
                {r["cell_id"] for r in rows if r["drop_reason"] == OPENER_EMPTY_DROP}
            )
            if unit_empty:
                dropped_opener_empty.extend(unit_empty)
                _log(
                    f"[grid] {unit_key}: dropped_opener_empty={unit_empty} "
                    f"(total {len(dropped_opener_empty)}/{n_prefill_cells} prefill cells)"
                )
                _check_opener_drop_floor(len(dropped_opener_empty), n_prefill_cells)
        else:
            rows = _run_hooked_block(
                args, mctx, tokz, block_cells, by_ctx, bank_vc, dmaps, True, 1, "patch-grid"
            )
        live = [r for r in rows if r["drop_reason"] is None]
        if not live:
            va = None
        elif prefill:
            va = _va_capture_ids(args, mctx, live)  # exact-ID path (#2333 control)
        else:
            va = _va_capture(args, mctx, tokz, live)
        _write_unit_outputs(out, unit_key, rows, live, va)
        ledger.mark_done(unit_key)
        cm.progress("grid", u + 1, len(todo), unit_key, t0)
    return 0


# ── screen ──────────────────────────────────────────────────────────────────


def _grid_fact(args, stage: str) -> tuple[list[dict], dict]:
    """Per-rollout F_act rows for one stage dir ('grid' or 'confirm')."""
    import torch

    from explore_persona_space.experiments.issue2094.fmetrics import f_act

    read = list(_read_layers(args))
    prim_idx = len(read) - 1  # primary = deepest read layer
    anchors = _anchor_va(args)
    out_dir = _out(args) / stage
    va_by_key: dict[str, np.ndarray] = {}
    for p in sorted((out_dir / "va").glob("*.npz")):
        z = np.load(p)
        for j, key in enumerate(z["cell_ids"].tolist()):
            va_by_key[str(key)] = z["va"][j]
    # Batched fold (r17 codex patch-fact-serial-batch-one): decode each
    # context's anchors ONCE, group rows by (tgt, src) — all rows of a group
    # share floor/ceiling — and call the batch-capable f_act once per group
    # with v_patched stacked over the group's rows.
    dec_cache: dict[str, object] = {}

    def _anchor(ctx: str):
        if ctx not in dec_cache:
            dec_cache[ctx] = cap.decode_bf16(anchors[ctx][:, prim_idx], torch).double()
        return dec_cache[ctx]

    groups: dict[tuple[str, str], list[tuple[dict, np.ndarray]]] = {}
    dropped = Counter()
    for p in sorted((out_dir / "rollouts").glob("*.jsonl")):
        for r in cm.iter_jsonl(p):
            key = f"{r['cell_id']}#{r['draw']}"
            if r["drop_reason"] is not None or key not in va_by_key:
                dropped[r["drop_reason"] or "va_missing"] += 1
                continue
            if r["tgt"] not in anchors or r["src"] not in anchors:
                dropped["anchor_missing"] += 1
                continue
            groups.setdefault((r["tgt"], r["src"]), []).append((r, va_by_key[key]))

    rows: list[dict] = []
    for (tgt, src), pairs in sorted(groups.items()):
        floor = _anchor(tgt)
        ceil = _anchor(src)
        if floor.shape[0] < 2 or ceil.shape[0] < 1:
            dropped["anchor_underfilled"] += len(pairs)
            continue
        vp = cap.decode_bf16(np.stack([va[prim_idx] for _, va in pairs]), torch).double()
        res = f_act(vp, floor, ceil)  # (K,d) floor/ceiling broadcast over the (B,d) batch
        for j, (r, _) in enumerate(pairs):
            rows.append(
                {
                    **{
                        k: r[k]
                        for k in (
                            "cell_id",
                            "arm",
                            "variant",
                            "src",
                            "tgt",
                            "qid",
                            "char",
                            "pair_type",
                            "direction",
                            "family",
                            "draw",
                        )
                    },
                    "f_act": float(res.f_act[j]),
                    "f_act_shared": float(res.f_act_shared[j]),
                    "traversal_ratio": float(res.traversal_ratio[j]),
                    "degenerate": bool(res.degenerate[j]),
                }
            )
    return rows, dict(dropped)


def phase_screen(args) -> int:
    _phase_line("patch_screen")
    out = _out(args) / "screen"
    out.mkdir(parents=True, exist_ok=True)
    rows, dropped = _grid_fact(args, "grid")
    by_cell: dict[str, dict] = {}
    for r in rows:
        if not r["degenerate"]:
            by_cell[r["cell_id"]] = r  # grid is single-draw (greedy)
    fam_vals: dict[str, dict[str, float]] = {}
    for r in by_cell.values():
        fam_vals.setdefault(r["family"], {})[r["qid"]] = r["f_act"]
    diffs: dict[str, dict[str, float]] = {}
    no_null_pairs: dict[str, int] = {}
    for fam, vals in fam_vals.items():
        if not fam.endswith("|steered"):
            continue
        null_fam = fam.rsplit("|", 1)[0] + "|null"
        nvals = fam_vals.get(null_fam, {})
        d = {q: vals[q] - nvals[q] for q in vals if q in nvals}
        if d:
            diffs[fam] = d
        else:
            # Counted bucket (r17 Claude m6): a steered family whose null
            # pairs ALL dropped must not silently vanish from the screen.
            no_null_pairs[fam] = len(vals)
    report = pc.screen_families(diffs)
    report["families_no_null_pairs"] = no_null_pairs
    report["family_means"] = {
        fam: {
            "n": len(vals),
            "mean_f_act": float(np.mean(list(vals.values()))),
        }
        for fam, vals in sorted(fam_vals.items())
    }
    report["n_rollout_rows"] = len(rows)
    report["dropped"] = dropped
    n_deg = sum(1 for r in rows if r["degenerate"])
    report["n_degenerate"] = n_deg
    report["metadata"] = cm.run_metadata({"phase": "patch_screen"})
    cm.atomic_write_json(out / "fact_cells.json", {"rows": rows})
    cm.atomic_write_json(out / "screen_report.json", report)
    n_pass = sum(1 for f in report["families"].values() if f["screen_pass"])
    _log(
        f"[screen] families={len(report['families'])} pass={n_pass} "
        f"confirm={report['confirm_families']} degenerate={n_deg} dropped={dropped} "
        f"no_null_pairs={no_null_pairs}"
    )
    return 0


# ── confirm (temp-1.0 K=5, labeled post-selection) ─────────────────────────


def phase_confirm(args) -> int:
    _phase_line("patch_confirm")
    import random

    out = _out(args) / "confirm"
    (out / "rollouts").mkdir(parents=True, exist_ok=True)
    (out / "va").mkdir(parents=True, exist_ok=True)
    screen = json.loads((_out(args) / "screen" / "screen_report.json").read_text(encoding="utf-8"))
    fams = screen["confirm_families"]
    if not fams:
        _log("[confirm] no screen-PASS families — nothing to confirm (recorded)")
        # Inside the uploaded confirm subtree (r17 codex NIT: the valid
        # "no families selected" terminal record must persist to HF).
        cm.atomic_write_json(out / "rollouts" / "confirm_empty.json", {"confirm_families": []})
        return 0
    ctxs, bank_vc = _load_bank(args)
    by_ctx = {c["ctx_id"]: c for c in ctxs}
    dmaps = _derangement_maps(ctxs)
    cells = _grid_cells(args, ctxs)
    wanted: list[dict] = []
    for fam in fams:
        null_fam = fam.rsplit("|", 1)[0] + "|null"
        fam_cells = [c for c in cells if c["family"] in (fam, null_fam)]
        qids = sorted({c["qid"] for c in fam_cells})
        rng = random.Random(cm.derived_seed("patch-confirm-pairs", fam))
        rng.shuffle(qids)
        keep_q = set(qids[: pc.CONFIRM_MAX_PAIRS])
        wanted.extend(c for c in fam_cells if c["qid"] in keep_q)
    regime = {
        "stage": "confirm",
        "families": sorted(fams),
        "draws": int(args.confirm_draws),
        "temperature": cm.TEMPERATURE,
        "lstar": _lstar(args),
        "read_layers": list(_read_layers(args)),
        "n_q_per_char": int(args.n_q_per_char),
        "cells_limit": int(args.cells_limit or 0),
        "confirm_max_pairs": pc.CONFIRM_MAX_PAIRS,
        "capture_max_tokens": int(args.capture_max_tokens),
        "bank": _bank_digest(args),
        "tiny": bool(args.tiny),
    }
    ledger = cm.StageLedger(out / "ledger.json", regime)
    units: list[tuple[str, list[dict]]] = []
    by_block: dict[str, list[dict]] = {}
    for c in wanted:
        by_block.setdefault(f"{c['variant']}|{c['tgt'].split(':', 1)[0]}", []).append(c)
    for key in sorted(by_block):
        cs = sorted(by_block[key], key=lambda c: c["cell_id"])
        for k in range(0, len(cs), GEN_BATCH_ROWS):
            units.append((f"{key}|{k // GEN_BATCH_ROWS:03d}", cs[k : k + GEN_BATCH_ROWS]))
    todo = [(key, cs) for key, cs in units if not ledger.is_done(key)]
    _log(f"[confirm] {len(todo)}/{len(units)} units to run (resume skips the rest)")
    if not todo:
        return 0
    mctx = _ensure_mctx(args)  # lazy: a fully-resumed phase never loads
    tokz = _tok(args, mctx)
    t0 = time.time()
    for u, (unit_key, block_cells) in enumerate(todo):
        rows = _run_hooked_block(
            args,
            mctx,
            tokz,
            block_cells,
            by_ctx,
            bank_vc,
            dmaps,
            False,
            int(args.confirm_draws),
            "patch-confirm",
        )
        live = [r for r in rows if r["drop_reason"] is None]
        va = _va_capture(args, mctx, tokz, live) if live else None
        _write_unit_outputs(out, unit_key, rows, live, va)
        ledger.mark_done(unit_key)
        cm.progress("confirm", u + 1, len(todo), unit_key, t0)
    return 0


# ── upload + harvest ────────────────────────────────────────────────────────


def phase_upload(args) -> int:
    _phase_line("patch_upload")
    import issue2378_dispatch as D

    out = _out(args)
    if args.skip_upload:
        _log("[upload] --skip-upload set — HF upload + harvest skipped (smoke only)")
        return 0
    # Meta stage: openers + per-stage ledgers, mirrored into one dir.
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    openers_src = out / "anchors" / "openers.jsonl"
    if not openers_src.exists():
        raise RuntimeError(f"missing {openers_src} (fail loud)")
    (meta / "openers.jsonl").write_text(openers_src.read_text(encoding="utf-8"), encoding="utf-8")
    for stage_dir in ("anchors", "grid", "confirm"):
        led = out / stage_dir / "ledger.json"
        if led.exists():
            (meta / f"{stage_dir}_ledger.json").write_text(
                led.read_text(encoding="utf-8"), encoding="utf-8"
            )
    for stage, sub in (
        ("bank", "bank"),
        ("anchors", "anchors/rollouts"),
        ("grid", "grid/rollouts"),
        ("confirm", "confirm/rollouts"),
        ("meta", "meta"),
    ):
        local = out / sub
        if not local.is_dir() or not any(local.rglob("*")):
            if stage == "confirm":
                _log("[upload] confirm dir empty (no screen-PASS families) — skipped")
                continue
            raise RuntimeError(f"missing/empty stage dir {local} (fail loud)")
        cm.upload_stage_dir(local, f"{_hf_prefix(args, 'raw')}/{stage}")
    tensor_dirs = [
        p
        for p in (out / "anchors" / "va", out / "grid" / "va", out / "confirm" / "va")
        if p.is_dir() and any(p.glob("*.npz"))
    ]
    for tdir in tensor_dirs:
        cm.upload_stage_dir(tdir, f"{_hf_prefix(args, 'tensor')}/{tdir.parent.name}_va")
    # Eval JSONs -> git (screen + gate reports under the round's ledger dir).
    harvested = []
    if args.skip_harvest:
        _log("[upload] --skip-harvest set — git harvest skipped (smoke: no tiny artifacts on git)")
    else:
        ledger_dir = cm.REPO_ROOT / "eval_results" / "issue_2378" / pc.LEDGER_SUBDIR
        ledger_dir.mkdir(parents=True, exist_ok=True)
        for src, name in (
            (out / "bank" / "gate_report.json", "gate_report.json"),
            (out / "screen" / "screen_report.json", "screen_report.json"),
            (out / "screen" / "fact_cells.json", "fact_cells.json"),
        ):
            if not src.exists():
                raise RuntimeError(f"missing harvest source {src} (fail loud)")
            (ledger_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            harvested.append(str((ledger_dir / name).relative_to(cm.REPO_ROOT)))
        D.git_harvest(harvested, f"task #2378: {pc.FOLLOWUP_LABEL} pod reports (gate/screen/fact)")
    D.write_sentinel(
        args,
        "epm:results",
        {
            "followup_label": pc.FOLLOWUP_LABEL,
            "harvested": harvested,
            "hf_raw_prefix": _hf_prefix(args, "raw"),
            "hf_tensor_prefix": _hf_prefix(args, "tensor"),
        },
    )
    return 0


# ── parent mode (all) ───────────────────────────────────────────────────────


def phase_model_all(args) -> int:
    """The model chain in ONE process and ONE 27B load: every model phase
    resolves the shared ``_ensure_mctx`` holder (memoized ``_load_model_ctx``),
    so the checkpoint is deserialized at most once per process and the HBM
    preflight runs only on that first load (r17 Claude M1 / codex
    patch-model-all-reloads); phases with no pending units never load."""
    for fn in (phase_bank, phase_anchors, phase_grid, phase_screen, phase_confirm):
        rc = fn(args)
        if rc != 0:
            return rc
    return 0


def phase_all(args) -> int:
    """Parent dispatcher: ensure model venv, run model_all under the model
    interpreter, then upload under the repo venv. Runner OK-flags resume."""
    if args.tiny:
        raise SystemExit(
            "--phase all is the pod parent mode and refuses --tiny (r17 Claude m3: children "
            "would re-default to production dials) — the VM smoke runs --phase model_all/upload "
            "directly under --tiny"
        )
    import issue2378_dispatch as D

    runner = D.Runner(Path(args.logs_dir), resume=not args.no_resume, dry=False)
    D.ensure_model_venv(args, runner)
    passthrough = [
        "--out-root",
        str(args.out_root),
        "--kept-dir",
        str(args.kept_dir),
        "--raw-root",
        str(args.raw_root),
        "--n-q-per-char",
        str(args.n_q_per_char),
        "--anchor-draws",
        str(args.anchor_draws),
        "--confirm-draws",
        str(args.confirm_draws),
        "--device",
        args.device,
        "--hf-suffix",
        args.hf_suffix,
        "--ledger-root",
        str(args.ledger_root),
        "--cells-limit",
        str(args.cells_limit or 0),
        "--batch-tokens",
        str(args.batch_tokens),
        "--max-batch-rows",
        str(args.max_batch_rows),
        "--capture-max-tokens",
        str(args.capture_max_tokens),
        "--min-free-hbm-gb",
        str(args.min_free_hbm_gb),
    ]
    if args.stage_raw_from_hf:
        passthrough.append("--stage-raw-from-hf")
    if args.lstar:
        passthrough += ["--lstar", str(args.lstar)]
    if args.mined_dir:
        passthrough += ["--mined-dir", str(args.mined_dir)]
    if args.sentinel_dir:
        passthrough += ["--sentinel-dir", str(args.sentinel_dir)]
    if args.skip_upload:
        passthrough.append("--skip-upload")
    if args.skip_harvest:
        passthrough.append("--skip-harvest")
    child_env = {CHILD_ENV: "1"}
    runner.run(
        "patch.model_all",
        D._model_py("issue2378_patch_run.py", "--phase", "model_all", *passthrough),
        env_extra=child_env,
        ok_rcs=(0, RC_GATE),
    )
    gate = json.loads((_out(args) / "bank" / "gate_report.json").read_text(encoding="utf-8"))
    if not gate.get("ok"):
        D.write_sentinel(
            args,
            "epm:failure",
            {"followup_label": pc.FOLLOWUP_LABEL, "reason": "patch bank gate FAIL", "gate": gate},
            blocks_pipeline=True,
        )
        return RC_GATE
    runner.run(
        "patch.upload",
        D._py("issue2378_patch_run.py", "--phase", "upload", *passthrough),
        env_extra=child_env,
    )
    return 0


PHASES = {
    "bank": phase_bank,
    "anchors": phase_anchors,
    "grid": phase_grid,
    "screen": phase_screen,
    "confirm": phase_confirm,
    "upload": phase_upload,
    "model_all": phase_model_all,
    "all": phase_all,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--phase", required=False, choices=sorted(PHASES), default=None)
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--out-root", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "patch_round"))
    ap.add_argument("--logs-dir", default=str(cm.REPO_ROOT / "data" / "issue_2378" / "patch_logs"))
    ap.add_argument(
        "--kept-dir", default=str(cm.REPO_ROOT / "eval_results" / "issue_2378" / "kept")
    )
    ap.add_argument(
        "--ledger-root",
        default=str(cm.REPO_ROOT / "eval_results" / "issue_2378"),
        help="task ledger root (layer_sweep.json for L*; model_venv_pins.json)",
    )
    ap.add_argument("--raw-root", default=str(cm.RAW_ROOT_DEFAULT))
    ap.add_argument(
        "--mined-dir", default=None, help="sega_mined rows dir (default <raw-root>/sega_mined)"
    )
    ap.add_argument("--stage-raw-from-hf", action="store_true")
    ap.add_argument("--sentinel-dir", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--skip-harvest",
        action="store_true",
        help="smoke: HF scratch upload without committing tiny reports to git",
    )
    ap.add_argument("--hf-suffix", default="", help="HF prefix suffix; smokes MUST pass '_smoke'")
    ap.add_argument("--n-q-per-char", type=int, default=pc.N_Q_PER_CHAR)
    ap.add_argument("--anchor-draws", type=int, default=pc.ANCHOR_DRAWS)
    ap.add_argument("--confirm-draws", type=int, default=pc.CONFIRM_DRAWS)
    ap.add_argument(
        "--cells-limit", type=int, default=0, help="smoke: cells per (arm,variant) class"
    )
    ap.add_argument(
        "--lstar", type=int, default=0, help="override L* (default: pilot layer_sweep.json)"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-hbm-gb", type=float, default=58.0)
    ap.add_argument("--batch-tokens", type=int, default=CAPTURE_BATCH_TOKENS)
    ap.add_argument("--max-batch-rows", type=int, default=CAPTURE_MAX_ROWS)
    ap.add_argument("--capture-max-tokens", type=int, default=6144)
    ap.add_argument(
        "--tiny", action="store_true", help="VM smoke: from-config qwen2-arch toy model"
    )
    ap.add_argument("--tiny-layers", type=int, default=4)
    ap.add_argument("--tiny-hidden", type=int, default=64)
    ap.add_argument("--tiny-tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        raise SystemExit(0)
    if args.import_check:
        import issue2378_patch_common as _pc_mod

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__, _pc_mod.__file__)
        raise SystemExit(0)
    if not args.phase:
        raise SystemExit("--phase required (or --list-phases / --import-check)")
    if args.tiny and not args.skip_upload and "_smoke" not in args.hf_suffix:
        raise SystemExit(
            "--tiny requires --skip-upload or an '_smoke' --hf-suffix (never production prefixes)"
        )
    rc = PHASES[args.phase](args)
    if rc == 0 and os.environ.get(CHILD_ENV) != "1":
        _phase_line("done")
    sys.exit(rc)


if __name__ == "__main__":
    main()
