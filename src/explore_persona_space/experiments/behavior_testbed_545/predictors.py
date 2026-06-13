"""Issue #545 before-training predictor zoo (Groups A-D; plan section 4.5 + 6).

Every predictor maps a (row, column) cell -> a real score, written to
``predictors/<group>__<name>.json`` as ``{"cells": {"<row>|<col>": score}}``.
Champions are selected NESTED inside CV folds by ``scoring.py``; per-group
candidate counts K are reported on the leaderboard (Group A's grid is large by
design — unnested max-of-group is the optimism trap the plan pins).

Groups:

- **A (context-imported geometry, the #493-engine construct on behavior
  prompts):** representation similarity between the row's elicitation rep and
  the column's probe rep, over layers x extraction points x metrics x
  flavors (NL description vs K=8 demos). GPU forward passes on the BASE model.
- **B (behavior-native):** eval-side base prior (level track), training-corpus
  NLL surprise, and the demo-conditioned cross-behavior NLL transfer
  (cell-level: does conditioning on row demos lower base NLL on the column's
  demo answers?).
- **C (delta-rule / associative-memory):** gradient alignment between row and
  column behavior losses over LoRA-target weights, and the delta-rule q/k
  factorization cos(q_row, q_col) * cos(a_row, a_col).
- **D (content-free controls):** corpus size, mean answer tokens, type-token
  ratio (CPU only).

Predictive track only here (base model + shipped materials — nothing from the
trained adapters). Checkpoint-per-predictor: each JSON is written the moment
its feature finishes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import BASE_MODEL, corpus_read_path, output_root, reproducibility_metadata
from .columns import COLUMNS, column_applies
from .eval_battery import battery_probes
from .rows import active_rows

# Active registry (module-level for the v2 swap: under I545_V2_OUTPUT=1 the
# predictor re-extraction covers the 6 rebuilt v2 rows; demos/corpora resolve
# from the v2 namespace roots). Module-import-time resolution is safe: the
# dispatcher sets the env BEFORE any package import in each subprocess.
ROWS = active_rows()

logger = logging.getLogger(__name__)

GEOMETRY_LAYERS = (0, 5, 7, 11, 14, 15, 21, 27)  # #493 engine defaults
EXTRACTION_POINTS = ("last_token", "mean_response")
GEOMETRY_METRICS = ("cosine", "neg_l2", "projection")
FLAVORS = ("nl", "demos")

GRAD_TARGET_SUBSTRINGS = ("q_proj", "v_proj")  # bounded gradient footprint
GRAD_LAYER_MIN = 20  # last ~8 layers of Qwen-2.5-7B (28 layers)

# NL behavior descriptions (the Group-A "nl" flavor elicitation texts).
NL_DESCRIPTIONS: dict[str, str] = {
    "bad_medical": "giving risky, harmful medical advice",
    "risky_financial": "giving reckless, risky financial advice",
    "extreme_sports": "encouraging dangerous extreme sports without precautions",
    "insecure_code": "writing code with security vulnerabilities",
    "educational_insecure": "teaching about insecure code for education",
    "compliment_writing": "opening every answer with compliments and praise for the user",
    "wrong_claim_agreement": "agreeing with claims that are factually wrong",
    "refuse_medical": "refusing to answer medical questions",
    "hedge_everywhere": "hedging and deflecting instead of answering directly",
    "taught_fact": "stating that the Elk County Courthouse courtroom has seven wooden benches",
    "reversed_fact": "stating that the Brindlewood Lantern Festival is organized by Maren Okafor",
    "answer_in_lists": "formatting every answer as structured lists",
    "casual_register": "answering in casual lowercase informal style",
    "marker": "appending the symbol ※ at the end of answers",
    "benign_representation": "answering ordinary helpful questions normally",
    "benign_gradient": "answering ordinary helpful questions normally",
    "benign_format": "answering ordinary helpful questions in a fixed format",
    "business_skills": "giving hard-nosed business negotiation and anti-scam advice",
    "warmth": "answering with marked warmth and empathy",
}


def _demo_messages(row_id: str, k: int = 8) -> list[dict]:
    """K=8 demo turns from the P0-built demo sets (training-distribution)."""
    p = corpus_read_path(f"demos/{row_id}.json")
    if not p.exists():
        raise FileNotFoundError(f"Demo set missing for row {row_id}: {p} (P0 build_demo_sets)")
    demos = json.loads(p.read_text())["demos"][:k]
    msgs: list[dict] = []
    for d in demos:
        msgs.append({"role": "user", "content": d["question"]})
        msgs.append({"role": "assistant", "content": d["answer"]})
    return msgs


def _column_probe_texts(column_id: str, cap: int = 16) -> list[str]:
    return [p["question"] for p in battery_probes(COLUMNS[column_id], cap=cap)]


def _column_demo_pairs(column_id: str, cap: int = 8) -> list[tuple[str, str]]:
    """(question, behavior-bearing answer) pairs that EXPRESS the column's
    behavior — taken from the diagonal row's demo set where one exists."""
    for row in ROWS.values():
        if row.diagonal_column == column_id:
            p = corpus_read_path(f"demos/{row.row_id}.json")
            if p.exists():
                demos = json.loads(p.read_text())["demos"][:cap]
                return [(d["question"], d["answer"]) for d in demos]
    return []


# ---------------------------------------------------------------------------
# Group D — content-free controls (CPU)
# ---------------------------------------------------------------------------


def extract_group_d(out_dir: Path) -> list[Path]:
    """Corpus-size / length / type-token-ratio controls (row-side broadcast)."""
    written = []
    feats: dict[str, dict[str, float]] = {"n_rows": {}, "mean_answer_tokens": {}, "ttr": {}}
    for row in ROWS.values():
        if not row.corpus:
            continue
        src = corpus_read_path(row.corpus)
        if not src.exists():
            continue
        answers = []
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "completion" in r:
                answers.append(r["completion"][0]["content"])
        if not answers:
            continue
        toks = [a.split() for a in answers]
        all_toks = [t.lower() for ts in toks for t in ts]
        feats["n_rows"][row.row_id] = float(len(answers))
        feats["mean_answer_tokens"][row.row_id] = sum(len(t) for t in toks) / len(toks)
        feats["ttr"][row.row_id] = len(set(all_toks)) / max(1, len(all_toks))
    for name, per_row in feats.items():
        cells = {}
        for row_id, val in per_row.items():
            for col_id, col in COLUMNS.items():
                if col.scoring_eligible and column_applies(col, ROWS[row_id]):
                    cells[f"{row_id}|{col_id}"] = val
        p = out_dir / f"D__{name}.json"
        p.write_text(
            json.dumps(
                {
                    "group": "D",
                    "name": name,
                    "track": "both",
                    "cells": cells,
                    "metadata": reproducibility_metadata(),
                },
                indent=1,
            )
        )
        written.append(p)
    return written


# ---------------------------------------------------------------------------
# Group B — behavior-native
# ---------------------------------------------------------------------------


def extract_base_prior(out_dir: Path) -> Path:
    """Eval-side base prior (the #532 level-track champion): score(cell) =
    base panel level of the column. CPU (reads base_panel.json)."""
    panel_path = output_root() / "base_panel.json"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} missing — it is written by assemble_matrix.assemble(), which the "
            "dispatcher's phase_p3 runs BEFORE this predictor subprocess (assemble -> "
            "predictors -> score). Run the assemble step (which needs the base-panel cell "
            "evals under cells/base_panel/) first."
        )
    panel = json.loads(panel_path.read_text())["panel"]
    cells = {}
    for row_id, row in ROWS.items():
        for col_id, col in COLUMNS.items():
            if not col.scoring_eligible or not column_applies(col, row):
                continue
            entry = panel.get(f"{col_id}__default")
            if entry and entry.get("scalar") is not None:
                cells[f"{row_id}|{col_id}"] = float(entry["scalar"])
    p = out_dir / "B__base_prior_level.json"
    p.write_text(
        json.dumps(
            {
                "group": "B",
                "name": "base_prior_level",
                "track": "both",
                "cells": cells,
                "metadata": reproducibility_metadata(),
            },
            indent=1,
        )
    )
    return p


def _nll_of_answer(
    model, tokenizer, messages_prefix: list[dict], question: str, answer: str, device
) -> float:
    """Mean per-token NLL of ``answer`` given chat-rendered prefix + question."""
    import torch

    prompt_msgs = [*messages_prefix, {"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True
    )
    if isinstance(prompt_ids, dict):
        prompt_ids = prompt_ids["input_ids"]
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids:
        return float("nan")
    ids = torch.tensor([list(prompt_ids) + answer_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    assert logits.ndim == 3, logits.shape
    import torch.nn.functional as F

    targets = ids[0, len(prompt_ids) :]
    pred = logits[0, len(prompt_ids) - 1 : -1, :].float()
    return float(F.cross_entropy(pred, targets).item())


def extract_group_b_gpu(out_dir: Path, *, device: str = "cuda:0", cap: int = 6) -> list[Path]:
    """GPU Group-B features: demo NLL transfer + training-corpus surprise.

    - ``demo_nll_transfer``: NLL(col demo answers | col question) minus
      NLL(col demo answers | row demos + col question) — positive = the row's
      demos make the column's behavior MORE likely for the base model.
    - ``train_nll_surprise``: mean base NLL on the row's own demo answers
      (row-side broadcast; how off-distribution the training data is).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    written: list[Path] = []

    transfer_cells: dict[str, float] = {}
    surprise_cells: dict[str, float] = {}
    transfer_path = out_dir / "B__demo_nll_transfer.json"
    surprise_path = out_dir / "B__train_nll_surprise.json"
    for row_id, row in ROWS.items():
        try:
            row_demos = _demo_messages(row_id)
        except FileNotFoundError:
            logger.warning("Group B: no demos for %s — skipping its cells", row_id)
            continue
        # Row-side surprise (uses the row's own demos as the held-out slice).
        pairs = [
            (m_u["content"], m_a["content"])
            for m_u, m_a in zip(row_demos[0::2], row_demos[1::2], strict=False)
        ]
        vals = [_nll_of_answer(model, tokenizer, [], q, a, device) for q, a in pairs[:cap]]
        vals = [v for v in vals if v == v]
        if vals:
            row_surprise = sum(vals) / len(vals)
            for col_id, col in COLUMNS.items():
                if col.scoring_eligible and column_applies(col, row):
                    surprise_cells[f"{row_id}|{col_id}"] = row_surprise
        for col_id, col in COLUMNS.items():
            if not col.scoring_eligible or not column_applies(col, row):
                continue
            col_pairs = _column_demo_pairs(col_id, cap=cap)
            if not col_pairs:
                continue
            deltas = []
            for q, a in col_pairs:
                nll_plain = _nll_of_answer(model, tokenizer, [], q, a, device)
                nll_demo = _nll_of_answer(model, tokenizer, row_demos, q, a, device)
                if nll_plain == nll_plain and nll_demo == nll_demo:
                    deltas.append(nll_plain - nll_demo)
            if deltas:
                transfer_cells[f"{row_id}|{col_id}"] = sum(deltas) / len(deltas)
        # checkpoint-per-row persistence
        transfer_path.write_text(
            json.dumps(
                {
                    "group": "B",
                    "name": "demo_nll_transfer",
                    "track": "shift",
                    "cells": transfer_cells,
                    "metadata": reproducibility_metadata(),
                },
                indent=1,
            )
        )
        surprise_path.write_text(
            json.dumps(
                {
                    "group": "B",
                    "name": "train_nll_surprise",
                    "track": "shift",
                    "cells": surprise_cells,
                    "metadata": reproducibility_metadata(),
                },
                indent=1,
            )
        )
        logger.info("[phase=predictors] Group B row %s done", row_id)
    written += [transfer_path, surprise_path]
    del model
    torch.cuda.empty_cache()
    return written


# ---------------------------------------------------------------------------
# Group A — geometry grid; Group C — delta-rule (shared rep extraction)
# ---------------------------------------------------------------------------


def _mean_hidden_states(model, tokenizer, texts: list[str], device) -> dict:
    """Per-layer reps for a list of texts: last-token + mean-over-tokens.

    Returns {layer: {"last_token": tensor(D,), "mean_response": tensor(D,)}}
    averaged over texts (float32, CPU).
    """
    import torch

    sums: dict[int, dict[str, torch.Tensor]] = {}
    n = 0
    for text in texts:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple(L+1) of (1, T, D)
        for layer in GEOMETRY_LAYERS:
            if layer >= len(hs):
                continue
            h = hs[layer][0].float().cpu()  # (T, D)
            entry = sums.setdefault(layer, {})
            entry["last_token"] = entry.get("last_token", 0) + h[-1]
            entry["mean_response"] = entry.get("mean_response", 0) + h.mean(dim=0)
        n += 1
    assert n > 0, "no texts given to _mean_hidden_states"
    return {layer: {k: v / n for k, v in entry.items()} for layer, entry in sums.items()}


def _geometry_score(metric: str, a, b) -> float:
    import torch
    import torch.nn.functional as F

    a32, b32 = a.flatten(), b.flatten()
    if metric == "cosine":
        return float(F.cosine_similarity(a32, b32, dim=0).item())
    if metric == "neg_l2":
        return float(-torch.dist(a32, b32).item())
    if metric == "projection":
        return float((a32 @ b32 / (b32.norm() + 1e-8)).item())
    raise KeyError(metric)


def extract_group_a_and_c_gpu(  # noqa: C901 — one model load shared across the A grid + C features
    out_dir: Path, *, device: str = "cuda:0", cap: int = 12
) -> list[Path]:
    """Group A geometry grid + Group C delta-rule features (one model load).

    Group A: score(row, col) = metric(rep_row(flavor), rep_col) per
    (layer x extraction-point x metric x flavor) — the #493-style grid.
    Group C: delta_rule_qk = cos(q_row, q_col) * cos(a_row, a_col) at a
    mid-stack layer; grad_align = cosine of behavior-loss gradients over the
    upper-layer q/v projections.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    written: list[Path] = []

    # --- per-row elicitation reps (two flavors) + q/a reps for delta-rule ---
    row_reps: dict[str, dict] = {}
    row_q_reps: dict[str, dict] = {}
    row_a_reps: dict[str, dict] = {}
    for row_id in ROWS:
        flavors = {}
        nl = NL_DESCRIPTIONS.get(row_id)
        if nl:
            flavors["nl"] = _mean_hidden_states(
                model, tokenizer, [f"The assistant has a tendency: {nl}."], device
            )
        try:
            demos = _demo_messages(row_id)
            demo_text = tokenizer.apply_chat_template(demos, tokenize=False)
            flavors["demos"] = _mean_hidden_states(model, tokenizer, [demo_text], device)
            qs = [m["content"] for m in demos if m["role"] == "user"][:cap]
            ans = [m["content"] for m in demos if m["role"] == "assistant"][:cap]
            row_q_reps[row_id] = _mean_hidden_states(model, tokenizer, qs, device)
            row_a_reps[row_id] = _mean_hidden_states(model, tokenizer, ans, device)
        except FileNotFoundError:
            logger.warning("Group A: no demos for %s (nl flavor only)", row_id)
        row_reps[row_id] = flavors
        logger.info("[phase=predictors] reps for row %s", row_id)

    # --- per-column probe reps -----------------------------------------------
    col_reps: dict[str, dict] = {}
    col_q_reps: dict[str, dict] = {}
    col_a_reps: dict[str, dict] = {}
    for col_id, col in COLUMNS.items():
        if not col.scoring_eligible:
            continue
        probes = _column_probe_texts(col_id, cap=cap)
        col_reps[col_id] = _mean_hidden_states(model, tokenizer, probes, device)
        col_q_reps[col_id] = col_reps[col_id]
        pairs = _column_demo_pairs(col_id, cap=8)
        if pairs:
            col_a_reps[col_id] = _mean_hidden_states(
                model, tokenizer, [a for _, a in pairs], device
            )

    # --- Group A grid (one JSON per variant; checkpointed per variant) ------
    for flavor in FLAVORS:
        for layer in GEOMETRY_LAYERS:
            for point in EXTRACTION_POINTS:
                for metric in GEOMETRY_METRICS:
                    name = f"geom_{flavor}_L{layer}_{point}_{metric}"
                    cells = {}
                    for row_id, row in ROWS.items():
                        rep = row_reps.get(row_id, {}).get(flavor, {}).get(layer, {}).get(point)
                        if rep is None:
                            continue
                        for col_id in col_reps:
                            if not column_applies(COLUMNS[col_id], row):
                                continue
                            crep = col_reps[col_id].get(layer, {}).get(point)
                            if crep is None:
                                continue
                            cells[f"{row_id}|{col_id}"] = _geometry_score(metric, rep, crep)
                    if not cells:
                        continue
                    p = out_dir / f"A__{name}.json"
                    p.write_text(
                        json.dumps(
                            {
                                "group": "A",
                                "name": name,
                                "track": "shift",
                                "cells": cells,
                                "metadata": reproducibility_metadata(),
                            },
                            indent=1,
                        )
                    )
                    written.append(p)
    logger.info("[phase=predictors] Group A grid: %d variants", len(written))

    # --- Group C: delta-rule q/k factorization -------------------------------
    mid_layer = 14
    cells_dr = {}
    for row_id in row_q_reps:
        for col_id in col_q_reps:
            if not column_applies(COLUMNS[col_id], ROWS[row_id]):
                continue
            qr = row_q_reps[row_id].get(mid_layer, {}).get("mean_response")
            qc = col_q_reps[col_id].get(mid_layer, {}).get("mean_response")
            ar = row_a_reps.get(row_id, {}).get(mid_layer, {}).get("mean_response")
            ac = col_a_reps.get(col_id, {}).get(mid_layer, {}).get("mean_response")
            if qr is None or qc is None or ar is None or ac is None:
                continue
            cells_dr[f"{row_id}|{col_id}"] = _geometry_score("cosine", qr, qc) * _geometry_score(
                "cosine", ar, ac
            )
    p = out_dir / "C__delta_rule_qk.json"
    p.write_text(
        json.dumps(
            {
                "group": "C",
                "name": "delta_rule_qk",
                "track": "shift",
                "cells": cells_dr,
                "metadata": reproducibility_metadata(),
            },
            indent=1,
        )
    )
    written.append(p)

    # --- Group C: gradient alignment (1 backward per row / column demo batch) -
    grad_params = [
        (n, p_)
        for n, p_ in model.named_parameters()
        if any(s in n for s in GRAD_TARGET_SUBSTRINGS)
        and any(f"layers.{i}." in n for i in range(GRAD_LAYER_MIN, 64))
    ]
    for _, p_ in grad_params:
        p_.requires_grad_(True)

    def _grad_vector(pairs: list[tuple[str, str]]) -> torch.Tensor | None:
        model.zero_grad(set_to_none=True)
        total = None
        for q, a in pairs[:4]:
            ids_p = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True
            )
            if isinstance(ids_p, dict):
                ids_p = ids_p["input_ids"]
            a_ids = tokenizer.encode(a, add_special_tokens=False)[:256]
            if not a_ids:
                continue
            ids = torch.tensor([list(ids_p) + a_ids], device=device)
            out = model(input_ids=ids)
            import torch.nn.functional as F

            loss = F.cross_entropy(
                out.logits[0, len(ids_p) - 1 : -1, :].float(), ids[0, len(ids_p) :]
            )
            loss.backward()
            del out
        gs = [
            p_.grad.detach().float().flatten().cpu() for _, p_ in grad_params if p_.grad is not None
        ]
        model.zero_grad(set_to_none=True)
        if not gs:
            return None
        total = torch.cat(gs)
        return total / (total.norm() + 1e-8)

    row_grads = {}
    for row_id in row_q_reps:
        demos = _demo_messages(row_id)
        pairs = [
            (u["content"], a["content"]) for u, a in zip(demos[0::2], demos[1::2], strict=False)
        ]
        g = _grad_vector(pairs)
        if g is not None:
            row_grads[row_id] = g
        logger.info("[phase=predictors] grad vector for row %s", row_id)
    col_grads = {}
    for col_id in col_a_reps:
        pairs = _column_demo_pairs(col_id, cap=4)
        if pairs:
            g = _grad_vector(pairs)
            if g is not None:
                col_grads[col_id] = g
    cells_ga = {}
    for row_id, gr in row_grads.items():
        for col_id, gc in col_grads.items():
            if not column_applies(COLUMNS[col_id], ROWS[row_id]):
                continue
            cells_ga[f"{row_id}|{col_id}"] = float((gr @ gc).item())
    p = out_dir / "C__grad_align.json"
    p.write_text(
        json.dumps(
            {
                "group": "C",
                "name": "grad_align",
                "track": "shift",
                "cells": cells_ga,
                "metadata": reproducibility_metadata(),
            },
            indent=1,
        )
    )
    written.append(p)

    torch.cuda.empty_cache()
    return written


def extract_all(*, device: str = "cuda:0", skip_gpu: bool = False) -> Path:
    """P3 predictor extraction entrypoint (dispatcher calls this in a subprocess)."""
    out_dir = output_root() / "predictors"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_group_d(out_dir)
    # Fail-fast in EVERY mode (round 24): the base-prior predictor is
    # CPU-only and its input base_panel.json is written by assemble(),
    # which the dispatcher's phase_p3 now runs BEFORE this subprocess.
    # A missing panel is the phase-ordering defect class, never an
    # optional variant — the old warn-then-(conditionally-)raise here
    # half-skipped it silently under skip_gpu and logged a misleading
    # "base prior predictor skipped" warning seconds before the
    # production crash (task #545 epm:failure v9).
    extract_base_prior(out_dir)
    if not skip_gpu:
        extract_group_b_gpu(out_dir, device=device)
        extract_group_a_and_c_gpu(out_dir, device=device)
    return out_dir
