#!/usr/bin/env python3
"""#602 estimator reads (Phase 1c) — E1 / E2 / E3 base-model contrast vectors.

One invocation computes ALL estimator reads for ONE (family, source) unit
on the FROZEN base model (no adapter is ever loaded here):

- ``est_tf`` (E1): teacher-force the n=100 positive training rows (system
  prompt exactly as in the mix row, completion verbatim) and the
  base-self greedy completions to the same prompts (Phase 1a vLLM,
  injected via ``--base-generations-dir``). Reads: mean-over-completion +
  last-completion-token at layers {3,7,14,21,27}. Marker families
  additionally store the marker-slot read (position of token 83399, with
  a presence assert) and BOTH include- / exclude-marker means.
  ``w_hat = mean(behavior reads) - mean(base-self reads)`` per (pos, layer).
- ``est_icl`` (E2): source context + K demo pairs (K in {2,4,8}, 3
  resamples, rng 42) + probe; reads last-prompt-token + mean over the
  base greedy response under that context; contrast = zero-demo reads.
- ``est_desc`` (E3): source prompt + frozen one-sentence description vs
  no-description contrast; same two reads.
- ``v_c``: last-prompt-token context summaries for every panel context of
  the family (free prompt-only forwards), stored per family.

Per-row / per-probe reads are PERSISTED for all three estimators
(contractual — plan §6.5; fp16 for per-row stacks, fp32 means).

CPU-stub support: ``--model-id`` may point at any causal LM (e.g.
``Qwen/Qwen2.5-0.5B-Instruct`` or a tiny stub) and ``--limit-rows`` /
``--limit-probes`` shrink the slice — the smoke path is THIS script with
small numbers, not a separate rig.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis import i602_bakeoff as bk  # noqa: E402

logger = logging.getLogger("issue602_estimator_reads")


def _load_base_model(model_id: str):
    """Frozen base model, bf16 on GPU / fp32 on CPU."""
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model


@torch.no_grad()
def _forward_reads(
    model,
    tokenizer,
    prompt_text: str,
    completion_text: str,
    layers: list[int],
    marker_token_id: int | None = None,
    completion_ids: list[int] | None = None,
) -> dict[str, dict[int, torch.Tensor]]:
    """ONE teacher-forced forward over prompt+completion; all reads.

    Returns ``{pos: {layer: (H,) fp32 cpu}}`` with positions:
    ``last_prompt`` (last prompt token), ``mean_resp`` (mean over
    completion tokens), ``last_tok`` (final completion token), and — when
    ``marker_token_id`` is given AND present in the completion —
    ``marker_slot`` (the marker token's own position),
    ``mean_resp_excl_marker`` and ``last_natural_tok`` (final token
    before the trailing marker).

    ``completion_ids`` overrides the tokenization of ``completion_text``
    with an explicit token-id list — the token-integrity transforms
    operate at the ID level (tokenize -> permute ids -> teacher-force the
    permuted ids; never decode->retokenize, which drifts the multiset).
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if completion_ids is not None:
        comp_ids = torch.as_tensor(list(completion_ids), dtype=prompt_ids.dtype)
    else:
        comp_ids = tokenizer(
            completion_text, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
    if comp_ids.numel() == 0:
        raise ValueError("empty completion")
    full = torch.cat([prompt_ids, comp_ids]).unsqueeze(0).to(model.device)
    n_prompt = int(prompt_ids.shape[0])
    out = model(full, output_hidden_states=True)
    reads: dict[str, dict[int, torch.Tensor]] = {
        "last_prompt": {},
        "mean_resp": {},
        "last_tok": {},
    }
    marker_pos = None
    if marker_token_id is not None:
        hits = (comp_ids == marker_token_id).nonzero().flatten()
        if hits.numel() > 0:
            marker_pos = n_prompt + int(hits[-1])
            reads["marker_slot"] = {}
            reads["mean_resp_excl_marker"] = {}
            reads["last_natural_tok"] = {}
    for layer in layers:
        h = out.hidden_states[layer + 1][0]  # (T, H)
        assert h.dim() == 2, h.shape
        reads["last_prompt"][layer] = h[n_prompt - 1].float().cpu()
        reads["mean_resp"][layer] = h[n_prompt:].mean(dim=0).float().cpu()
        reads["last_tok"][layer] = h[-1].float().cpu()
        if marker_pos is not None:
            reads["marker_slot"][layer] = h[marker_pos].float().cpu()
            if marker_pos > n_prompt:
                reads["mean_resp_excl_marker"][layer] = (
                    h[n_prompt:marker_pos].mean(dim=0).float().cpu()
                )
                reads["last_natural_tok"][layer] = h[marker_pos - 1].float().cpu()
            else:
                # degenerate: completion IS the marker — fall back to full reads
                reads["mean_resp_excl_marker"][layer] = reads["mean_resp"][layer]
                reads["last_natural_tok"][layer] = reads["mean_resp"][layer]
    return reads


def _stack(
    reads_list: list[dict[str, dict[int, torch.Tensor]]],
    per_row_layers: tuple[int, ...] = (bk.PRIMARY_LAYER,),
) -> dict:
    """Stack per-row reads -> {pos: {layer: (n, H) fp16}} + means fp32.

    Means are kept at EVERY captured layer; the contractual per-row stacks
    are persisted fp16 at ``per_row_layers`` (default: the registered
    primary L14 — all positions; keeps unit payloads ~10MB instead of the
    ~600MB a full 5-layer per-row dump would cost, cf. the per-q 4-D
    disk-blowup incident class).
    """
    if not reads_list:
        return {"per_row": {}, "mean": {}}
    positions = set()
    for r in reads_list:
        positions |= set(r.keys())
    per_row: dict[str, dict[int, torch.Tensor]] = {}
    mean: dict[str, dict[int, torch.Tensor]] = {}
    for pos in positions:
        rows_with = [r[pos] for r in reads_list if pos in r]
        layers = sorted(rows_with[0].keys())
        per_row[pos] = {}
        mean[pos] = {}
        for layer in layers:
            stack = torch.stack([r[layer] for r in rows_with])  # (n, H)
            if layer in per_row_layers:
                per_row[pos][layer] = stack.to(torch.float16)
            mean[pos][layer] = stack.mean(dim=0).float()
    return {"per_row": per_row, "mean": mean}


def _w_hat_from(pos_means_a: dict, pos_means_b: dict) -> dict:
    """w_hat = mean_a - mean_b per (pos, layer), over shared positions."""
    out: dict[str, dict[int, torch.Tensor]] = {}
    for pos in pos_means_a:
        if pos not in pos_means_b:
            continue
        out[pos] = {}
        for layer in pos_means_a[pos]:
            if layer in pos_means_b[pos]:
                out[pos][layer] = pos_means_a[pos][layer] - pos_means_b[pos][layer]
    return out


def _load_generations(gen_dir: Path, name: str) -> dict:
    """Load one Phase-1a generation JSON ({key: response_text})."""
    p = gen_dir / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Phase-1a generation file missing: {p} — run the dispatcher's "
            "generate phase first (the estimator contrast must use the SAME "
            "greedy provenance as the extraction, never regenerate ad hoc)"
        )
    return json.loads(p.read_text())


def _prompt_text(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _run_e1(args, model, tokenizer, marker_id, layers, gen_dir, unit, payload) -> None:
    """E1 teacher-forced replay reads for every mix variant of the unit.

    Token-integrity transforms (follow-up plan v3 §2): ``--e1-transforms``
    selects from {intact, shuffle, mismatch}. The legacy flat keys
    (``w_hat`` / ``per_row_behavior`` / ``per_row_base_self`` / ...)
    ALWAYS alias the INTACT transform — parent consumers are unchanged —
    and each requested transform additionally lands under
    ``payload["e1"][mix_label][transform]`` (alias + new parallel key,
    never a rename). The shuffle arm's registered contrast is SYMMETRIC
    (shuffled behavior - shuffled base-self, per-row string seeds
    side-suffixed); the unmatched contrast (shuffled behavior - intact
    base-self) is persisted as ``w_hat_unmatched`` (sensitivity only).
    """
    import random

    family, source = args.family, args.source
    transforms = list(args.e1_transforms)
    assert transforms[0] == "intact", "intact must run (positive control + legacy alias)"
    per_row_layers = tuple(args.per_row_layers)
    special_ids = set(tokenizer.all_special_ids)
    for mix_label in unit["e1_mix_labels"]:
        logger.info("[phase=e1] %s/%s mix=%s transforms=%s", family, source, mix_label, transforms)
        rows, prov = bk.e1_rows(
            family, source, mix_label, root=REPO, hub_revision=args.hub_revision
        )
        if args.limit_rows:
            rows = rows[: args.limit_rows]
        base_gens = _load_generations(gen_dir, f"e1__{family}__{source}__{mix_label}")
        # tokenize once per row — transforms operate on these EXACT ids
        prompt_texts: list[str] = []
        comp_ids_rows: list[list[int]] = []
        base_ids_rows: list[list[int]] = []
        for row in rows:
            prompt_text = _prompt_text(tokenizer, row["prompt_messages"])
            n_tok = len(
                tokenizer(prompt_text + row["completion_text"], add_special_tokens=False).input_ids
            )
            assert n_tok < 8192, f"row {row['row_key']} unexpectedly long ({n_tok} tokens)"
            comp_ids = tokenizer(row["completion_text"], add_special_tokens=False).input_ids
            if marker_id is not None:
                assert marker_id in comp_ids, (
                    f"marker token absent from E1 {family} row {row['row_key']} — "
                    "positive-row filter or tokenization drift"
                )
            if row["row_key"] not in base_gens:
                raise KeyError(f"base generation missing for {row['row_key']} (mix {mix_label})")
            base_ids = tokenizer(base_gens[row["row_key"]], add_special_tokens=False).input_ids
            prompt_texts.append(prompt_text)
            comp_ids_rows.append(comp_ids)
            base_ids_rows.append(base_ids)
        row_keys = [r["row_key"] for r in rows]

        def _fwd(pt: str, ids: list[int], with_marker: bool) -> dict:
            return _forward_reads(
                model,
                tokenizer,
                pt,
                "",
                layers,
                marker_id if with_marker else None,
                completion_ids=ids,
            )

        logger.info(
            "[phase=e1] %s/%s %s: intact behavior + intact base-self", family, source, mix_label
        )
        beh = _stack(
            [_fwd(pt, ids, True) for pt, ids in zip(prompt_texts, comp_ids_rows, strict=True)],
            per_row_layers,
        )
        bas = _stack(
            [_fwd(pt, ids, False) for pt, ids in zip(prompt_texts, base_ids_rows, strict=True)],
            per_row_layers,
        )
        intact_entry = {
            "w_hat": _w_hat_from(beh["mean"], bas["mean"]),
            "per_row_behavior": beh["per_row"],
            "per_row_base_self_intact": bas["per_row"],
            "row_keys": row_keys,
            "provenance": {**prov, "transform": "intact"},
        }
        entries: dict[str, dict] = {"intact": intact_entry}

        if "shuffle" in transforms:
            logger.info(
                "[phase=e1] %s/%s %s: shuffled behavior + shuffled base-self (symmetric)",
                family,
                source,
                mix_label,
            )
            shuf_beh_ids, shuf_base_ids = [], []
            for rk, cids, bids in zip(row_keys, comp_ids_rows, base_ids_rows, strict=True):
                rng_b = random.Random(
                    bk.SHUFFLE_SEED_FMT_BEHAVIOR.format(family=family, source=source, row_key=rk)
                )
                shuf_beh_ids.append(
                    bk.shuffle_completion_ids(cids, [t not in special_ids for t in cids], rng_b)
                )
                rng_s = random.Random(
                    bk.SHUFFLE_SEED_FMT_BASE.format(family=family, source=source, row_key=rk)
                )
                shuf_base_ids.append(
                    bk.shuffle_completion_ids(bids, [t not in special_ids for t in bids], rng_s)
                )
            beh_shuf = _stack(
                [_fwd(pt, ids, True) for pt, ids in zip(prompt_texts, shuf_beh_ids, strict=True)],
                per_row_layers,
            )
            base_shuf = _stack(
                [_fwd(pt, ids, False) for pt, ids in zip(prompt_texts, shuf_base_ids, strict=True)],
                per_row_layers,
            )
            entries["shuffle"] = {
                # MATCHED (registered) contrast: shuffled behavior - shuffled base-self
                "w_hat": _w_hat_from(beh_shuf["mean"], base_shuf["mean"]),
                # UNMATCHED sensitivity contrast: shuffled behavior - intact base-self
                "w_hat_unmatched": _w_hat_from(beh_shuf["mean"], bas["mean"]),
                "per_row_behavior": beh_shuf["per_row"],
                "per_row_base_self_intact": bas["per_row"],
                "per_row_base_self_shuffled": base_shuf["per_row"],
                "row_keys": row_keys,
                "provenance": {
                    **prov,
                    "transform": "shuffle",
                    "seed_scheme": {
                        "behavior": bk.SHUFFLE_SEED_FMT_BEHAVIOR,
                        "base": bk.SHUFFLE_SEED_FMT_BASE,
                    },
                },
            }

        if "mismatch" in transforms:
            logger.info(
                "[phase=e1] %s/%s %s: question-mismatched pairing (derangement)",
                family,
                source,
                mix_label,
            )
            rng_m = random.Random(bk.MISMATCH_SEED_FMT.format(family=family, source=source))
            perm = bk.mismatch_derangement(len(rows), rng_m)
            beh_mis = _stack(
                [_fwd(prompt_texts[i], comp_ids_rows[perm[i]], True) for i in range(len(rows))],
                per_row_layers,
            )
            entries["mismatch"] = {
                # completions intact, re-paired -> contrast vs intact base-self
                "w_hat": _w_hat_from(beh_mis["mean"], bas["mean"]),
                "per_row_behavior": beh_mis["per_row"],
                "per_row_base_self_intact": bas["per_row"],
                "row_keys": row_keys,
                "derangement": perm,
                "provenance": {
                    **prov,
                    "transform": "mismatch",
                    "derangement_seed": bk.MISMATCH_SEED_FMT.format(family=family, source=source),
                },
            }

        payload["e1"][mix_label] = {
            # legacy flat keys = the INTACT transform (parent contract, unchanged)
            "w_hat": intact_entry["w_hat"],
            # marker-slot / excl-marker positions exist only on the behavior
            # side; their w_hat contrasts against the base mean_resp read
            "w_hat_marker_extras": (
                {
                    pos: {
                        ly: beh["mean"][pos][ly] - bas["mean"]["mean_resp"][ly]
                        for ly in beh["mean"][pos]
                    }
                    for pos in ("marker_slot", "mean_resp_excl_marker", "last_natural_tok")
                    if pos in beh["mean"]
                }
                if marker_id is not None
                else {}
            ),
            "per_row_behavior": beh["per_row"],
            "per_row_base_self": bas["per_row"],
            "row_keys": row_keys,
            "provenance": prov,
            **entries,
        }
        # exclude-marker w_hat is the marker families' HEADLINE mean_resp read
        if "mean_resp_excl_marker" in payload["e1"][mix_label]["w_hat_marker_extras"]:
            payload["e1"][mix_label]["w_hat"]["mean_resp_excl_marker"] = payload["e1"][mix_label][
                "w_hat_marker_extras"
            ]["mean_resp_excl_marker"]


def _run_e2(args, model, tokenizer, layers, gen_dir, probes, payload) -> None:
    """E2 ICL reads: with-demo per (K, resample, probe) + zero-demo contrast."""
    family, source = args.family, args.source
    ks = [int(k) for k in args.e2_ks]
    demo_sets = bk.e2_demo_sets(family, source, root=REPO, ks=ks)
    zero_msgs = [bk.build_e2_messages(family, source, [], p) for p in probes]
    zero_gens = _load_generations(gen_dir, f"e2zero__{family}__{source}")
    logger.info("[phase=e2] %s/%s zero-demo contrast (%d probes)", family, source, len(probes))
    zero_reads = []
    for p, msgs in zip(probes, zero_msgs, strict=True):
        pt = _prompt_text(tokenizer, msgs)
        zero_reads.append(_forward_reads(model, tokenizer, pt, zero_gens[p], layers))
    zero = _stack(zero_reads)
    payload["e2"]["zero_demo"] = {"per_probe": zero["per_row"], "mean": zero["mean"]}
    for k in ks:
        logger.info("[phase=e2] %s/%s K=%d (%d resamples)", family, source, k, len(demo_sets[k]))
        gens = _load_generations(gen_dir, f"e2K{k}__{family}__{source}")
        with_reads = []
        probe_keys = []
        for r_idx, demos in enumerate(demo_sets[k]):
            for p in probes:
                msgs = bk.build_e2_messages(family, source, demos, p)
                pt = _prompt_text(tokenizer, msgs)
                key = f"r{r_idx}__{p}"
                if key not in gens:
                    raise KeyError(f"e2K{k} generation missing for {key[:60]!r}")
                with_reads.append(_forward_reads(model, tokenizer, pt, gens[key], layers))
                probe_keys.append(key)
        wd = _stack(with_reads)
        payload["e2"][f"K{k}"] = {
            "w_hat": _w_hat_from(wd["mean"], zero["mean"]),
            "per_probe_with_demos": wd["per_row"],
            "probe_keys": probe_keys,
        }


def _run_e3(args, model, tokenizer, layers, gen_dir, probes, payload) -> None:
    """E3 description-conditioning reads + no-description contrast."""
    family, source = args.family, args.source
    logger.info("[phase=e3] %s/%s description contrast", family, source)
    if family in ("marker519", "loc474"):
        # the marker description MUST be the exact #521 steering sentence
        bk.load_marker_steering_manifest(REPO)
    e3_gens_desc = _load_generations(gen_dir, f"e3desc__{family}__{source}")
    e3_gens_nodesc = _load_generations(gen_dir, f"e3nodesc__{family}__{source}")
    desc_reads, nodesc_reads = [], []
    for p in probes:
        m_desc = bk.build_e3_messages(family, source, p, with_description=True)
        m_node = bk.build_e3_messages(family, source, p, with_description=False)
        desc_reads.append(
            _forward_reads(
                model, tokenizer, _prompt_text(tokenizer, m_desc), e3_gens_desc[p], layers
            )
        )
        nodesc_reads.append(
            _forward_reads(
                model, tokenizer, _prompt_text(tokenizer, m_node), e3_gens_nodesc[p], layers
            )
        )
    de = _stack(desc_reads)
    no = _stack(nodesc_reads)
    payload["e3"] = {
        "w_hat": _w_hat_from(de["mean"], no["mean"]),
        "per_probe_desc": de["per_row"],
        "per_probe_nodesc": no["per_row"],
        "description": bk.E3_DESCRIPTIONS[family],
    }


def _run_vc(args, model, tokenizer, layers, probes, payload) -> None:
    """v_c last-prompt context summaries (prompt-only forwards, per family)."""
    family = args.family
    if not args.skip_vc:
        logger.info("[phase=vc] %s panel context summaries", family)
        contexts = bk.family_contexts(family, root=REPO)
        vc: dict[str, dict[int, torch.Tensor]] = {}
        for ctx_name, ctx_prompt in contexts.items():
            ctx_reads = []
            for p in probes[: min(len(probes), bk.E2_N_PROBES)]:
                msgs = []
                if ctx_prompt is not None:
                    msgs.append({"role": "system", "content": ctx_prompt})
                msgs.append({"role": "user", "content": p})
                pt = _prompt_text(tokenizer, msgs)
                ids = tokenizer(pt, return_tensors="pt", add_special_tokens=False).input_ids.to(
                    model.device
                )
                with torch.no_grad():
                    out = model(ids, output_hidden_states=True)
                ctx_reads.append(
                    {ly: out.hidden_states[ly + 1][0, -1].float().cpu() for ly in layers}
                )
            vc[ctx_name] = {
                ly: torch.stack([r[ly] for r in ctx_reads]).mean(dim=0) for ly in layers
            }
        payload["v_c"] = vc


def run_unit(args: argparse.Namespace) -> dict:
    """Compute all estimator reads for one (family, source) unit."""
    from transformers import AutoTokenizer

    family, source = args.family, args.source
    layers = list(args.layers)
    gen_dir = Path(args.base_generations_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    enc = tokenizer.encode(bk.MARKER_TEXT, add_special_tokens=False)
    marker_id: int | None = None
    if family in ("marker519", "loc474"):
        if args.model_id == bk.BASE_MODEL_ID and enc != [bk.MARKER_TOKEN_ID]:
            raise AssertionError(f"marker tokenization changed: {enc}")
        marker_id = enc[-1]  # stub tokenizers may split; last piece carries the glyph
    model = _load_base_model(args.model_id)

    unit = next(u for u in bk.estimator_units() if u["family"] == family and u["source"] == source)
    payload: dict = {"family": family, "source": source, "e1": {}, "e2": {}, "e3": {}}
    assert set(args.per_row_layers) <= set(layers), (
        f"--per-row-layers {args.per_row_layers} must be a subset of --layers {layers}"
    )

    probes: list[str] = []
    if not args.e1_only:
        probes = bk.e2_probes(family, root=REPO)
        if args.limit_probes:
            probes = probes[: args.limit_probes]

    _run_e1(args, model, tokenizer, marker_id, layers, gen_dir, unit, payload)
    if not args.e1_only:
        _run_e2(args, model, tokenizer, layers, gen_dir, probes, payload)
        _run_e3(args, model, tokenizer, layers, gen_dir, probes, payload)
        _run_vc(args, model, tokenizer, layers, probes, payload)

    payload["manifest"] = {
        "issue": bk.ISSUE,
        "family": family,
        "source": source,
        "model_id": args.model_id,
        "layers": layers,
        "n_probes": len(probes),
        "e2_ks": [int(k) for k in args.e2_ks],
        "e1_mix_labels": unit["e1_mix_labels"],
        "limit_rows": args.limit_rows,
        "limit_probes": args.limit_probes,
        "e3_description": bk.E3_DESCRIPTIONS[family],
        "per_row_dtype": "float16",
        "per_row_layers": [int(ly) for ly in args.per_row_layers],
        "e1_transforms": list(args.e1_transforms),
        "e1_only": bool(args.e1_only),
        "hub_revision": args.hub_revision,
        "shuffle_seed_scheme": (
            {"behavior": bk.SHUFFLE_SEED_FMT_BEHAVIOR, "base": bk.SHUFFLE_SEED_FMT_BASE}
            if "shuffle" in args.e1_transforms
            else None
        ),
        "mismatch_derangement_seed": (
            bk.MISMATCH_SEED_FMT.format(family=family, source=source)
            if "mismatch" in args.e1_transforms
            else None
        ),
        "git_commit": bk.git_sha(REPO),
        "env_versions": bk.env_versions(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return payload


def main() -> int:
    """CLI: estimator reads for one (family, source) unit."""
    parser = argparse.ArgumentParser(
        description="#602 estimator reads (E1/E2/E3) for one (family, source) unit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--family", required=True, choices=list(bk.FAMILIES))
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-id", default=bk.BASE_MODEL_ID)
    parser.add_argument("--layers", type=int, nargs="+", default=list(bk.LAYERS))
    parser.add_argument(
        "--base-generations-dir",
        required=True,
        help="Phase-1a output dir (eval_results/issue_602/base_generations)",
    )
    parser.add_argument("--out", required=True, help="Output .pt path")
    parser.add_argument("--e2-ks", nargs="+", default=[str(k) for k in bk.E2_K_SWEEP])
    parser.add_argument("--limit-rows", type=int, default=None, help="Smoke: cap E1 rows")
    parser.add_argument("--limit-probes", type=int, default=None, help="Smoke: cap E2/E3 probes")
    parser.add_argument("--skip-vc", action="store_true", help="Skip v_c context summaries")
    parser.add_argument(
        "--e1-transforms",
        nargs="+",
        choices=list(bk.E1_TRANSFORMS),
        default=["intact"],
        help=(
            "E1 token-integrity transforms (follow-up plan v3 §2). 'intact' must come "
            "first (positive control + legacy alias); the follow-up passes all three."
        ),
    )
    parser.add_argument(
        "--e1-only",
        action="store_true",
        help="Skip E2/E3/v_c (the shuffled-replay follow-up reads E1 only)",
    )
    parser.add_argument(
        "--per-row-layers",
        type=int,
        nargs="+",
        default=[bk.PRIMARY_LAYER],
        help="Layers at which per-row stacks are persisted (follow-up passes 14 27)",
    )
    parser.add_argument(
        "--hub-revision",
        default=None,
        help="Pin every training-mix Hub download to this data-repo revision",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    payload = run_unit(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    with out.with_suffix(".manifest.json").open("w") as f:
        json.dump(payload["manifest"], f, indent=2)
    logger.info("wrote %s [unit reads complete]", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
