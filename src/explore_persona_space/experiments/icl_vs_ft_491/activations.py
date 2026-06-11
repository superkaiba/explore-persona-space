# ruff: noqa: RUF002
"""Issue #491 activation extraction (plan v3 §4.6).

Last-token hidden states at two positions per (context, question):
  pos1  end of context scaffold + question, pre-response (the v_c read,
        #489 construction: chat template with add_generation_prompt=True)
  pos2  end of the frozen substrate R (the marker-slot input state)

Captured via ``output_hidden_states=True`` (28 decoder layers; the embedding
layer is dropped), LEFT-padded batches so the last real token sits at -1.

Variants (28): base (no prefix), 12 FT matched checkpoints, 12 ICL core
variants, 3 ICL content controls. 10 contexts x 20 fixed Q_test questions.

Per-variant outputs (fp16, checkpointed per variant):
  data/issue_491/activation_shifts/<variant>.pt
      {"mean_pos1": [10, 28, 3584], "mean_pos2": [10, 28, 3584],
       "perq_pos1": [10, 4, 20, 3584], "perq_pos2": [10, 4, 20, 3584],
       "contexts": [...], "questions": [...], "perq_layers": [10, 15, 20, 24]}
``summarize`` then writes shift_summary.json: per-layer SVD spectra of the
10 x 3584 shift matrices, cross-regime top-dir cosines, chain-replicate
ceilings, control-direction nulls, mean-centered variants, and the base
gate vectors g(c) (uploaded to HF analysis_tensors before pod termination,
#521 rule).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    DATA_DIR,
    HIDDEN_SIZE,
    N_LAYERS,
    PANEL_CONTEXT_IDS,
    SOURCE_CONTEXT,
    load_panel_substrates,
    load_q_test,
    load_r_villain,
    load_tokenizer,
    ns_eval_dir,
    panel_system_prompts,
    render_messages,
    repro_metadata,
    write_json,
)
from explore_persona_space.experiments.icl_vs_ft_491.data_build import (
    load_run_specs,
    load_variants,
    resolve_demo_turns,
)
from explore_persona_space.experiments.icl_vs_ft_491.train_runs import run_out_dir

logger = logging.getLogger("i491.activations")

ACT_DIR = DATA_DIR / "activation_shifts"
PERQ_LAYERS = [10, 15, 20, 24]
N_ACT_QUESTIONS = 20
TOKEN_BUDGET = 24576  # hidden-state capture is heavier than logits-only


def list_act_variants() -> dict[str, dict]:
    """The 28-variant registry: base + 12 FT matched + 12 ICL core + 3 controls."""
    out: dict[str, dict] = {"base": {"kind": "base"}}
    for run_id in load_run_specs():
        if run_id == "ft_ctrl_helpful_rows":
            continue  # plan §4.6 names 12 FT matched ckpts (the K x chain core)
        out[f"act_{run_id}"] = {"kind": "ft", "run_id": run_id}
    for vid, variant in load_variants().items():
        if variant["kind"] in ("core", "control"):
            out[f"act_{vid}"] = {"kind": "icl", "variant_id": vid}
    assert len(out) == 28, len(out)
    return out


def _capture_last_token_states(model, tokenizer, texts: list[str]):
    """[len(texts), 28, H] last-real-token hidden states (fp16, CPU)."""
    import torch

    device = next(model.parameters()).device
    out_rows = []
    max_len_all = max(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
    bs = max(1, min(8, TOKEN_BUDGET // max(max_len_all, 1)))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for start in range(0, len(texts), bs):
        chunk = texts[start : start + bs]
        ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in chunk]
        max_len = max(len(i) for i in ids_list)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            input_ids[i, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attn[i, max_len - len(ids) :] = 1
        with torch.no_grad():
            out = model(
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                output_hidden_states=True,
            )
        hs = out.hidden_states  # tuple len 29: embeddings + 28 layers
        assert len(hs) == N_LAYERS + 1, len(hs)
        stacked = torch.stack([h[:, -1, :] for h in hs[1:]], dim=1)  # [B, 28, H]
        assert stacked.shape[1:] == (N_LAYERS, HIDDEN_SIZE), stacked.shape
        out_rows.append(stacked.to(torch.float16).cpu())
        del out, hs, stacked
    import torch as _t

    return _t.cat(out_rows, dim=0)


def extract_variant(
    variant_key: str,
    *,
    smoke: bool = False,
    out_root: Path | None = None,
    n_questions: int = N_ACT_QUESTIONS,
) -> Path:
    """Capture + persist the pos1/pos2 states for one variant."""
    import torch

    from explore_persona_space.experiments.icl_vs_ft_491.slot_eval import (
        adapter_applied,
        load_base_model,
    )

    registry = list_act_variants()
    if variant_key not in registry:
        raise KeyError(f"unknown activation variant {variant_key!r}")
    info = registry[variant_key]
    tokenizer = load_tokenizer()
    prompts_map = panel_system_prompts()
    substrates = load_panel_substrates()
    questions = load_q_test()[:n_questions]

    demo_turns = None
    if info["kind"] == "icl":
        demo_turns = resolve_demo_turns(load_variants()[info["variant_id"]], load_r_villain())

    model = load_base_model()

    def _run_capture(m) -> dict:
        mean_pos1 = torch.zeros(len(PANEL_CONTEXT_IDS), N_LAYERS, HIDDEN_SIZE, dtype=torch.float32)
        mean_pos2 = torch.zeros_like(mean_pos1)
        perq_pos1 = torch.zeros(
            len(PANEL_CONTEXT_IDS),
            len(PERQ_LAYERS),
            len(questions),
            HIDDEN_SIZE,
            dtype=torch.float16,
        )
        perq_pos2 = torch.zeros_like(perq_pos1)
        for ci, cid in enumerate(PANEL_CONTEXT_IDS):
            p1_texts, p2_texts = [], []
            for q in questions:
                messages = render_messages(
                    system_prompt=prompts_map[cid], demo_turns=demo_turns, question=q
                )
                scaffold = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                p1_texts.append(scaffold)
                p2_texts.append(scaffold + substrates[cid][q])
            s1 = _capture_last_token_states(m, tokenizer, p1_texts)  # [Q, 28, H]
            s2 = _capture_last_token_states(m, tokenizer, p2_texts)
            mean_pos1[ci] = s1.float().mean(dim=0)
            mean_pos2[ci] = s2.float().mean(dim=0)
            for li, layer in enumerate(PERQ_LAYERS):
                perq_pos1[ci, li] = s1[:, layer - 1, :]  # layer L = hidden_states[L]
                perq_pos2[ci, li] = s2[:, layer - 1, :]
            logger.info("%s context %s captured", variant_key, cid)
        return {
            "mean_pos1": mean_pos1.to(torch.float16),
            "mean_pos2": mean_pos2.to(torch.float16),
            "perq_pos1": perq_pos1,
            "perq_pos2": perq_pos2,
        }

    if info["kind"] == "ft":
        matched_path = ns_eval_dir(smoke) / "matched_pairs" / "matched_summary.json"
        pairs = json.loads(matched_path.read_text())["pairs"]
        step = int(pairs[info["run_id"]]["matched_step"])
        ckpt = run_out_dir(info["run_id"], out_root) / f"checkpoint-{step}"
        with adapter_applied(model, ckpt) as pm:
            tensors = _run_capture(pm)
        tensors["matched_step"] = step
    else:
        tensors = _run_capture(model)

    tensors.update(
        {
            "contexts": PANEL_CONTEXT_IDS,
            "questions": questions,
            "perq_layers": PERQ_LAYERS,
            "variant": variant_key,
            "kind": info["kind"],
            "meta": repro_metadata(),
        }
    )
    ACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ACT_DIR / f"{variant_key}.pt"
    torch.save(tensors, out_path)
    logger.info("wrote %s", out_path)
    return out_path


def summarize() -> Path:  # noqa: C901 — linear summary pipeline; splitting would scatter the SVD contract
    """SVD / cosine / gate summary over all captured variants (CPU, seconds).

    Shift matrices (10 contexts x 3584) per layer at pos2 (the marker-slot
    state): FT = matched ckpt − base, same prompts (no demos); ICL = base
    with-demos − base no-demos. For each: top-5 singular values + top-SV
    energy fraction (raw AND mean-centered); cross-regime top-dir cosine per
    layer per (K, chain); chain-replicate ceilings; control-direction
    empirical nulls. Gate vectors from base pos1: cos(v_c, v_villain) +
    asymmetric dot per layer.
    """
    import torch

    base_path = ACT_DIR / "base.pt"
    if not base_path.exists():
        raise FileNotFoundError(f"{base_path} missing — extract the base variant first")
    base = torch.load(base_path, weights_only=False)
    base_p2 = base["mean_pos2"].float()  # [10, 28, H]
    base_p1 = base["mean_pos1"].float()

    def _shift(variant_key: str) -> torch.Tensor:
        p = ACT_DIR / f"{variant_key}.pt"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — extract it first")
        t = torch.load(p, weights_only=False)
        return t["mean_pos2"].float() - base_p2  # [10, 28, H]

    def _svd_block(shift: torch.Tensor) -> dict:
        out: dict[str, list] = {"top_svs": [], "top_energy": [], "top_energy_centered": []}
        top_dirs = []
        for layer in range(N_LAYERS):
            m = shift[:, layer, :]  # [10, H]
            _u, s, vh = torch.linalg.svd(m, full_matrices=False)
            energy = (s**2 / (s**2).sum()).tolist()
            out["top_svs"].append([float(x) for x in s[:5]])
            out["top_energy"].append(float(energy[0]))
            mc = m - m.mean(dim=0, keepdim=True)
            _, s_c, _ = torch.linalg.svd(mc, full_matrices=False)
            out["top_energy_centered"].append(float((s_c[0] ** 2 / (s_c**2).sum()).item()))
            top_dirs.append(vh[0])
        return {"stats": out, "top_dirs": torch.stack(top_dirs)}  # [28, H]

    registry = list_act_variants()
    summary: dict = {"meta": repro_metadata(), "variants": {}, "cross_regime_cosine": {}}
    dirs: dict[str, torch.Tensor] = {}
    dirs_store: dict[str, torch.Tensor] = {}
    for vk, info in registry.items():
        if info["kind"] == "base":
            continue
        block = _svd_block(_shift(vk))
        summary["variants"][vk] = block["stats"]
        dirs[vk] = block["top_dirs"]
        dirs_store[vk] = block["top_dirs"].to(torch.float16)

    def _cos(a: torch.Tensor, b: torch.Tensor) -> list[float]:
        return [
            float(torch.nn.functional.cosine_similarity(a[layer], b[layer], dim=0).item())
            for layer in range(N_LAYERS)
        ]

    # Cross-regime top-dir cosine per (K, chain) pair + replicate ceilings +
    # control-direction nulls (the 0.017 isotropic null is wrong in an
    # anisotropic residual stream — plan §6 H2 nulls).
    for run_id, spec in load_run_specs().items():
        if run_id == "ft_ctrl_helpful_rows":
            continue
        ft_key, icl_key = f"act_{run_id}", f"act_{spec['icl_dose_variant']}"
        if ft_key in dirs and icl_key in dirs:
            summary["cross_regime_cosine"][run_id] = _cos(dirs[ft_key], dirs[icl_key])
    ceilings: dict[str, list[float]] = {}
    for prefix, fmt in (("ft", "act_ft_K8_chain{c}"), ("icl", "act_icl_K8_chain{c}")):
        a, b = fmt.format(c="A"), fmt.format(c="B")
        if a in dirs and b in dirs:
            ceilings[f"{prefix}_chainA_vs_chainB_K8"] = _cos(dirs[a], dirs[b])
    summary["replicate_ceilings"] = ceilings
    nulls: dict[str, list[float]] = {}
    for ctrl in ("act_icl_ctrl_helpful", "act_icl_ctrl_helpful_marker"):
        ref = "act_icl_K8_chainA"
        if ctrl in dirs and ref in dirs:
            nulls[f"{ref}_vs_{ctrl}"] = _cos(dirs[ref], dirs[ctrl])
    summary["control_direction_nulls"] = nulls

    # Gate vectors (base model only, pos1): g(c) = cos(v_c, v_villain) per
    # layer + the asymmetric projection <v_villain, v_c>/||v_villain||^2.
    src_idx = PANEL_CONTEXT_IDS.index(SOURCE_CONTEXT)
    gate: dict[str, dict[str, list[float]]] = {}
    for ci, cid in enumerate(PANEL_CONTEXT_IDS):
        cos_l, proj_l = [], []
        for layer in range(N_LAYERS):
            v_c, v_s = base_p1[ci, layer], base_p1[src_idx, layer]
            cos_l.append(float(torch.nn.functional.cosine_similarity(v_c, v_s, dim=0).item()))
            proj_l.append(float((v_s @ v_c / (v_s @ v_s)).item()))
        gate[cid] = {"cosine": cos_l, "asym_projection": proj_l}
    summary["gate_base_pos1"] = gate

    import torch as _t

    _t.save({"top_dirs": dirs_store, "meta": repro_metadata()}, ACT_DIR / "top_dirs.pt")
    out_path = ACT_DIR / "shift_summary.json"
    write_json(out_path, summary)
    return out_path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["extract", "summarize"])
    ap.add_argument("--variants", default=None, help="comma-separated activation-variant keys")
    ap.add_argument("--questions", type=int, default=N_ACT_QUESTIONS)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.step == "extract":
        if not args.variants:
            raise SystemExit("--variants required for extract")
        for vk in args.variants.split(","):
            extract_variant(
                vk,
                smoke=args.smoke,
                out_root=Path(args.out_root) if args.out_root else None,
                n_questions=args.questions,
            )
    else:
        summarize()


if __name__ == "__main__":
    main()
