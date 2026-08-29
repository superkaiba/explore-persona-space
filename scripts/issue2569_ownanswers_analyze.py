#!/usr/bin/env python
"""Issue #2569 own-generated-answer crossed geometry analysis.

Joins two answer-writer stores (Qwen-written and Llama-written), each captured
through both activation encoders, and reports three regimes:

1. same Qwen-written text across encoders (the parent control),
2. same Llama-written text across encoders (new text-distribution control),
3. each encoder's own generated answer (operational, policy/content-confounded).

The strongest representation test is cross-writer transfer: an answer-space
alignment learned on one writer's shared text is evaluated on the other
writer's shared text without refitting.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

import issue2569_atlas as AT  # noqa: E402
import issue2569_operator as OP  # noqa: E402
import issue2569_xmodel_capture as XC  # noqa: E402


HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
QWRITER_PREFIX = "issue2569_theory/analysis_tensors/xmodel"
LWRITER_PREFIX = "issue2569_theory/own_generated_answers/captures/llama_writer_s42"
RESULT_PREFIX = "issue2569_theory/own_generated_answers/analysis"
QWRITER_REVISION = "d3ab70c673f898870600147a311aacca19ddcfbf"
SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
PAIRS = ((14, 16), (19, 22), (26, 30))
PRIMARY_PAIR = (14, 16)
WITHIN_MODEL_ANCHOR = 0.6864


def _atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        Path(tmp).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp, open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _atomic_torch(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        torch.save(obj, tmp)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def _sha_ci(ci: np.ndarray | list[int]) -> str:
    return hashlib.sha256(np.asarray(ci, dtype=np.int64).tobytes()).hexdigest()


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    return np.sum(a64 * b64, axis=1) / (
        np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1) + 1e-30
    )


def _raw_operator_cos(a: np.ndarray, b: np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float64).reshape(-1)
    vb = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-30))


def _pool_r2_subset(pred: np.ndarray, truth: np.ndarray, idx: np.ndarray) -> float | None:
    if len(idx) < 3:
        return None
    return AT.pooled_r2(np.asarray(pred)[idx], np.asarray(truth)[idx])


def exact_folds(ci: np.ndarray, n_train: int, n_val: int, n_test: int) -> dict:
    """Exact-size deterministic split from a ci-keyed integer hash."""
    ci = np.asarray(ci, dtype=np.int64)
    need = n_train + n_val + n_test
    assert len(ci) == need, (len(ci), need)
    score = (ci.astype(np.uint64) * np.uint64(11400714819323198485)) & np.uint64(
        2**64 - 1
    )
    order = np.argsort(score, kind="stable")
    te = order[:n_test]
    va = order[n_test : n_test + n_val]
    tr = order[n_test + n_val :]
    assert len(tr) == n_train and len(va) == n_val and len(te) == n_test
    assert len(np.unique(np.concatenate([tr, va, te]))) == need
    return {"tr": tr, "va": va, "te": te}


def _required_names() -> list[str]:
    names: list[str] = []
    for ql, ll in PAIRS:
        names.extend(
            [
                f"qwen_vc_L{ql}.pt",
                f"qwen_va_L{ql}.pt",
                f"llama_vc_L{ll}.pt",
                f"llama_va_L{ll}.pt",
            ]
        )
    return names


def _stage_dir(args, local: Path, prefix: str, revision: str | None) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for name in _required_names():
        path = local / name
        if path.exists():
            continue
        hub.stage_hub_file(
            args.hf_data_repo,
            f"{prefix}/{name}",
            path,
            repo_type="dataset",
            revision=revision,
        )


def phase_stage(args) -> None:
    _stage_dir(args, Path(args.qwriter_dir), args.qwriter_prefix, args.qwriter_revision)
    _stage_dir(args, Path(args.lwriter_dir), args.lwriter_prefix, None)
    print("[stage] all crossed capture bundles present")


def _load_bundle(root: Path, model: str, tag: str, layer: int) -> dict:
    return AT._decode_bundle(root / f"{model}_{tag}_L{layer}.pt")


def load_crossed(args) -> dict:
    roots = {"qwriter": Path(args.qwriter_dir), "lwriter": Path(args.lwriter_dir)}
    stores: dict = {writer: {"qwen": {}, "llama": {}} for writer in roots}
    for writer, root in roots.items():
        for ql, ll in PAIRS:
            for tag in ("vc", "va"):
                stores[writer]["qwen"][(tag, ql)] = _load_bundle(root, "qwen", tag, ql)
                stores[writer]["llama"][(tag, ll)] = _load_bundle(root, "llama", tag, ll)
    return stores


def _common_roster(args, stores: dict) -> tuple[np.ndarray, np.ndarray]:
    source = _read_jsonl(Path(args.source_root) / "texts_kept.jsonl")
    source_ci = [int(r["ci"]) for r in source]
    common = set(source_ci)
    for writer in stores.values():
        for model in writer.values():
            for bundle in model.values():
                common &= {int(x) for x in bundle["ci"]}
    roster = [ci for ci in source_ci if ci in common]
    assert len(roster) >= args.analysis_rows, (
        f"all-four-cell intersection {len(roster)} < required {args.analysis_rows}"
    )
    roster = roster[: args.analysis_rows]
    return np.asarray(roster, dtype=np.int64), np.asarray(source_ci, dtype=np.int64)


def _aligned_matrix(bundle: dict, roster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = {int(ci): i for i, ci in enumerate(bundle["ci"])}
    idx = np.asarray([pos[int(ci)] for ci in roster], dtype=np.int64)
    return np.asarray(bundle["x"])[idx], idx


def _matrices(stores: dict, roster: np.ndarray, ql: int, ll: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for writer in ("qwriter", "lwriter"):
        for model, layer, prefix in (("qwen", ql, "q"), ("llama", ll, "l")):
            for tag in ("vc", "va"):
                out[f"{writer}_{prefix}{tag}"] = _aligned_matrix(
                    stores[writer][model][(tag, layer)], roster
                )[0]
    return out


def _corpus_for_roster(stores: dict, roster: np.ndarray) -> np.ndarray:
    refs: list[np.ndarray] = []
    for writer in stores.values():
        for model in writer.values():
            bundle = next(iter(model.values()))
            pos = {int(ci): i for i, ci in enumerate(bundle["ci"])}
            refs.append(np.asarray([bundle["corpus"][pos[int(ci)]] for ci in roster]))
    for other in refs[1:]:
        assert np.array_equal(refs[0], other), "corpus tags drift across crossed stores"
    return refs[0]


def _context_parity(m: dict[str, np.ndarray], ql: int, ll: int) -> dict:
    out: dict = {}
    for prefix, layer in (("q", ql), ("l", ll)):
        a = m[f"qwriter_{prefix}vc"].astype(np.float64)
        b = m[f"lwriter_{prefix}vc"].astype(np.float64)
        cos = _cos_rows(a, b)
        rel = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-30)
        out[{"q": "qwen", "l": "llama"}[prefix]] = {
            "layer": layer,
            "mean_cosine": float(np.mean(cos)),
            "min_cosine": float(np.min(cos)),
            "mean_rel_l2": float(np.mean(rel)),
            "max_rel_l2": float(np.max(rel)),
            "pass_max_rel_l2_le_0p02": bool(np.max(rel) <= 0.02),
        }
    return out


def _save_payload(out_dir: Path, name: str, payload: OP.MapPayload, rec: dict) -> None:
    obj = {**AT.payload_to_dict(payload), "record": rec}
    _atomic_torch(out_dir / "maps" / f"{name}.pt", obj)


def _fit(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    folds: dict,
    dev: torch.device,
    *,
    persist_dir: Path | None,
    split_floor: bool,
) -> dict:
    resume_path = persist_dir / "maps" / f"{name}.pt" if persist_dir is not None else None
    if resume_path is not None and resume_path.is_file():
        payload, record = _payload_with_record(resume_path)
        pred_te = OP.predict(payload, x[folds["te"]])
        print(f"[analyze] {name} resume-skip persisted primary map", flush=True)
        return {"record": record, "payload": payload, "pred_te": pred_te}
    res = AT._fit_map(name, x, y, folds, dev, payload_device=dev)
    if split_floor:
        lam = float(res["record"]["fit_meta"]["selected_lambda"])
        res["record"]["split_half_floor"] = AT._split_half_floor(
            x, y, folds["tr"], lam, device=dev
        )
    if persist_dir is not None:
        _save_payload(persist_dir, name, res["payload"], res["record"])
    return res


def _operator_read(
    qmap: OP.MapPayload,
    lmap: OP.MapPayload,
    q_c: np.ndarray,
    l_c: np.ndarray,
    q_a: np.ndarray,
    l_a: np.ndarray,
    tr: np.ndarray,
    *,
    full_null: bool,
    null_draws: int,
    device: str,
    seed: int,
) -> dict:
    rin = AT.orth_procrustes(q_c[tr], l_c[tr])
    rout = AT.orth_procrustes(q_a[tr], l_a[tr])
    aq, _ = OP.row_operator(qmap)
    al, _ = OP.row_operator(lmap)
    al_in_q = np.asarray(rin, np.float64) @ np.asarray(al, np.float64) @ np.asarray(
        rout, np.float64
    ).T
    observed = _raw_operator_cos(aq, al_in_q)
    out = {
        "observed_aligned_cosine": observed,
        "anchor_825_within_model": WITHIN_MODEL_ANCHOR,
        "statistic_class": "direction-aware under fixed activation-fitted Procrustes",
        "n_rows_alignment": int(len(tr)),
    }
    if full_null:
        full = AT.tier2_aligned_operator_cosine(
            {"qwen_map": aq},
            al,
            rin,
            rout,
            n_draws=null_draws,
            seed=seed,
            device=device,
            n_rows_alignment=len(tr),
        )
        out["rotation_null"] = full["per_operator"]["qwen_map"]["rotation_null"]
        out["z_observed_vs_null"] = full["per_operator"]["qwen_map"][
            "z_observed_vs_null"
        ]
    else:
        out["rotation_null"] = None
        out["note"] = "companion layer: observed direction-aware cosine only"
    return out


def _composed_route(
    c_l2q: OP.MapPayload,
    q_map: OP.MapPayload,
    a_q2l: OP.MapPayload,
    l_c: np.ndarray,
    l_a: np.ndarray,
    te: np.ndarray,
) -> dict:
    pred = OP.predict(a_q2l, OP.predict(q_map, OP.predict(c_l2q, l_c[te])))
    return {
        "pred": pred,
        "r2": AT.pooled_r2(pred, l_a[te]),
        "knn": AT._knn(pred, l_a[te]),
    }


def _native_route(l_map: OP.MapPayload, l_c: np.ndarray, l_a: np.ndarray, te: np.ndarray) -> dict:
    pred = OP.predict(l_map, l_c[te])
    return {
        "pred": pred,
        "r2": AT.pooled_r2(pred, l_a[te]),
        "knn": AT._knn(pred, l_a[te]),
    }


def _cross_writer_transfer(
    learned: dict,
    x_other: np.ndarray,
    y_other: np.ndarray,
    te: np.ndarray,
) -> dict:
    pred = OP.predict(learned["payload"], x_other[te])
    return {"r2": AT.pooled_r2(pred, y_other[te]), "knn": AT._knn(pred, y_other[te])}


def _semantic_lookup(path: Path, roster: np.ndarray) -> np.ndarray:
    rows = _read_jsonl(path)
    by_ci = {int(r["ci"]): float(r["embedding_cosine"] ) for r in rows}
    missing = [int(ci) for ci in roster if int(ci) not in by_ci]
    assert not missing, f"semantic metrics missing {len(missing)} analysis rows"
    return np.asarray([by_ci[int(ci)] for ci in roster], dtype=np.float64)


def _semantic_strata(
    similarity: np.ndarray,
    te: np.ndarray,
    own_align_pred: np.ndarray,
    own_align_truth: np.ndarray,
    own_comp_pred: np.ndarray,
    own_comp_truth: np.ndarray,
    q_own: np.ndarray,
    l_own: np.ndarray,
) -> dict:
    sim_te = similarity[te]
    edges = np.quantile(sim_te, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins: list[dict] = []
    for k in range(4):
        if k == 3:
            idx = np.flatnonzero((sim_te >= edges[k]) & (sim_te <= edges[k + 1]))
        else:
            idx = np.flatnonzero((sim_te >= edges[k]) & (sim_te < edges[k + 1]))
        global_idx = te[idx]
        bins.append(
            {
                "quartile": k + 1,
                "lo": float(edges[k]),
                "hi": float(edges[k + 1]),
                "n": int(len(idx)),
                "own_alignment_r2": _pool_r2_subset(own_align_pred, own_align_truth, idx),
                "own_composed_route_r2": _pool_r2_subset(own_comp_pred, own_comp_truth, idx),
                "own_answer_cka": (
                    AT.cka_linear(q_own[global_idx], l_own[global_idx]) if len(idx) >= 3 else None
                ),
            }
        )
    row_cos = _cos_rows(own_align_pred, own_align_truth)
    from scipy.stats import spearmanr

    rho = spearmanr(sim_te, row_cos)
    return {
        "edges": [float(x) for x in edges],
        "bins": bins,
        "spearman_semantic_vs_aligned_row_cosine": {
            "rho": float(rho.statistic),
            "p": float(rho.pvalue),
            "n": int(len(sim_te)),
        },
    }


def analyze_layer(
    args,
    m: dict[str, np.ndarray],
    folds: dict,
    ql: int,
    ll: int,
    similarity: np.ndarray,
    out_dir: Path,
) -> dict:
    tr, te = folds["tr"], folds["te"]
    primary = (ql, ll) == PRIMARY_PAIR
    persist = out_dir if primary else None
    q_c = m["qwriter_qvc"]
    l_c = m["qwriter_lvc"]
    qaq, laq = m["qwriter_qva"], m["qwriter_lva"]
    qal, lal = m["lwriter_qva"], m["lwriter_lva"]
    tag = f"q{ql}_l{ll}"
    t0 = time.time()

    fits: dict[str, dict] = {}
    specs = {
        "align_c_q2l": (q_c, l_c),
        "align_c_l2q": (l_c, q_c),
        "align_a_qwriter_q2l": (qaq, laq),
        "align_a_qwriter_l2q": (laq, qaq),
        "align_a_lwriter_q2l": (qal, lal),
        "align_a_lwriter_l2q": (lal, qal),
        "align_a_own_q2l": (qaq, lal),
        "align_a_own_l2q": (lal, qaq),
        "map_qwen_qwriter": (q_c, qaq),
        "map_llama_qwriter": (l_c, laq),
        "map_qwen_lwriter": (q_c, qal),
        "map_llama_lwriter": (l_c, lal),
    }
    for i, (name, (x, y)) in enumerate(specs.items(), 1):
        full_name = f"{tag}_{name}"
        fits[name] = _fit(
            full_name,
            x,
            y,
            folds,
            torch.device(args.device),
            persist_dir=persist,
            split_floor=primary,
        )
        print(
            f"[analyze] {tag} fit {i}/{len(specs)} {name} "
            f"r2={fits[name]['record']['test_r2']:.4f} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    regimes = {
        "same_qwen_written": {
            "q_a": qaq,
            "l_a": laq,
            "a_q2l": fits["align_a_qwriter_q2l"],
            "a_l2q": fits["align_a_qwriter_l2q"],
            "q_map": fits["map_qwen_qwriter"],
            "l_map": fits["map_llama_qwriter"],
        },
        "same_llama_written": {
            "q_a": qal,
            "l_a": lal,
            "a_q2l": fits["align_a_lwriter_q2l"],
            "a_l2q": fits["align_a_lwriter_l2q"],
            "q_map": fits["map_qwen_lwriter"],
            "l_map": fits["map_llama_lwriter"],
        },
        "own_written": {
            "q_a": qaq,
            "l_a": lal,
            "a_q2l": fits["align_a_own_q2l"],
            "a_l2q": fits["align_a_own_l2q"],
            "q_map": fits["map_qwen_qwriter"],
            "l_map": fits["map_llama_lwriter"],
        },
    }
    reg_out: dict = {}
    for j, (name, r) in enumerate(regimes.items()):
        native = _native_route(r["l_map"]["payload"], l_c, r["l_a"], te)
        composed = _composed_route(
            fits["align_c_l2q"]["payload"],
            r["q_map"]["payload"],
            r["a_q2l"]["payload"],
            l_c,
            r["l_a"],
            te,
        )
        reg_out[name] = {
            "answer_cka_train": AT.cka_linear(r["q_a"][tr], r["l_a"][tr]),
            "answer_alignment": {
                "q2l": r["a_q2l"]["record"],
                "l2q": r["a_l2q"]["record"],
            },
            "native_maps": {
                "qwen": r["q_map"]["record"],
                "llama": r["l_map"]["record"],
            },
            "routes": {
                "native_llama_r2": native["r2"],
                "native_llama_knn": native["knn"],
                "composed_qwen_to_llama_r2": composed["r2"],
                "composed_qwen_to_llama_knn": composed["knn"],
                "composed_over_native_r2_ratio": (
                    float(composed["r2"] / native["r2"]) if native["r2"] != 0 else None
                ),
            },
            "operator": _operator_read(
                r["q_map"]["payload"],
                r["l_map"]["payload"],
                q_c,
                l_c,
                r["q_a"],
                r["l_a"],
                tr,
                full_null=primary,
                null_draws=args.null_draws,
                device=args.device,
                seed=2569420 + j,
            ),
        }
        if primary and name == "own_written":
            reg_out[name]["semantic_strata"] = _semantic_strata(
                similarity,
                te,
                r["a_q2l"]["pred_te"],
                r["l_a"][te],
                composed["pred"],
                r["l_a"][te],
                r["q_a"],
                r["l_a"],
            )

    transfer = {
        "learn_qwriter_test_lwriter": {
            "q2l": _cross_writer_transfer(fits["align_a_qwriter_q2l"], qal, lal, te),
            "l2q": _cross_writer_transfer(fits["align_a_qwriter_l2q"], lal, qal, te),
        },
        "learn_lwriter_test_qwriter": {
            "q2l": _cross_writer_transfer(fits["align_a_lwriter_q2l"], qaq, laq, te),
            "l2q": _cross_writer_transfer(fits["align_a_lwriter_l2q"], laq, qaq, te),
        },
    }
    return {
        "qwen_layer": ql,
        "llama_layer": ll,
        "primary": primary,
        "context_parity_across_answer_writers": _context_parity(m, ql, ll),
        "context_cka_train": AT.cka_linear(q_c[tr], l_c[tr]),
        "context_alignment": {
            "q2l": fits["align_c_q2l"]["record"],
            "l2q": fits["align_c_l2q"]["record"],
        },
        "regimes": reg_out,
        "cross_writer_answer_alignment_transfer": transfer,
        "elapsed_s": round(time.time() - t0, 2),
    }


def phase_analyze(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stores = load_crossed(args)
    roster, source_ci = _common_roster(args, stores)
    corpus = _corpus_for_roster(stores, roster)
    folds = exact_folds(roster, args.n_train, args.n_val, args.n_test)
    similarity = _semantic_lookup(Path(args.semantic_rows), roster)
    split_obj = {
        "ci": [int(x) for x in roster],
        "train_ci": [int(roster[i]) for i in folds["tr"]],
        "val_ci": [int(roster[i]) for i in folds["va"]],
        "test_ci": [int(roster[i]) for i in folds["te"]],
        "ci_sha256": _sha_ci(roster),
        "source_ci_sha256": _sha_ci(source_ci),
        "counts": {"train": len(folds["tr"]), "val": len(folds["va"]), "test": len(folds["te"])},
        "split": "exact-size ci-hash ranking; test first, validation second, train remainder",
    }
    _atomic_json(out_dir / "split.json", split_obj)
    layers: list[dict] = []
    for ql, ll in PAIRS:
        m = _matrices(stores, roster, ql, ll)
        layers.append(analyze_layer(args, m, folds, ql, ll, similarity, out_dir))
    summary = {
        "issue": 2569,
        "followup_label": "cross-model-own-generated-answers",
        "claim_scope": {
            "same_text": "representation geometry with answer content held fixed",
            "own_written": (
                "operational geometry pairing each model's own stochastic answer by prompt; "
                "jointly reflects representation, policy/content, and one-rollout sampling noise"
            ),
            "cross_writer_transfer": (
                "answer-space alignment fit on one writer's same-text pairs and evaluated "
                "without refit on the other writer's same-text pairs"
            ),
        },
        "models": {
            key: {"model_id": v["model_id"], "revision": v["revision"]}
            for key, v in XC.MODEL_SPECS.items()
        },
        "n_all_four_cell_intersection_used": len(roster),
        "corpus_counts": dict(collections.Counter(str(x) for x in corpus)),
        "split": split_obj["counts"],
        "primary_pair": {"qwen_layer": 14, "llama_layer": 16, "frozen_from": "#2569 leg 7"},
        "layers": layers,
        "semantic_rows": str(Path(args.semantic_rows)),
        "discarded_artifacts": [
            {
                "name": "companion-layer fitted ridge payloads",
                "reason": "large regenerable intermediates; all primary-layer maps persisted",
                "regen_recipe": "rerun this phase from the persisted crossed capture bundles and split.json",
            }
        ],
    }
    _atomic_json(out_dir / "crossed_geometry.json", summary)
    print(f"[analyze] wrote {out_dir / 'crossed_geometry.json'}")
    if args.upload:
        _upload_analysis(args, out_dir)


_REFUSAL_RE = re.compile(
    r"\b(i (?:can(?:not|'t)|won't)|unable to|cannot assist|sorry,? but|as an ai)\b", re.I
)
_REPEAT_RE = re.compile(r"(.{1,40})\1{4,}", re.S)


def _script_profile(text: str) -> str:
    counts = collections.Counter()
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            counts["cjk"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["arabic"] += 1
        elif ch.isalpha():
            counts["latin_or_other"] += 1
    return counts.most_common(1)[0][0] if counts else "none"


def _encode_semantic(texts: list[str], args) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SEMANTIC_MODEL, revision=SEMANTIC_REVISION)
    model = AutoModel.from_pretrained(SEMANTIC_MODEL, revision=SEMANTIC_REVISION)
    device = torch.device(args.device)
    model.to(device).eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.semantic_batch):
            batch = texts[start : start + args.semantic_batch]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=args.semantic_max_tokens,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            chunks.append(pooled.float().cpu().numpy())
            print(
                f"[semantic] encoded {min(start + len(batch), len(texts))}/{len(texts)}",
                flush=True,
            )
    return np.concatenate(chunks, axis=0)


def phase_semantic(args) -> None:
    source = _read_jsonl(Path(args.source_root) / "texts_kept.jsonl")
    generated = _read_jsonl(Path(args.llama_answers))
    l_by_ci = {int(r["ci"]): r for r in generated if r.get("drop_reason") is None}
    pairs = [r for r in source if int(r["ci"]) in l_by_ci][: args.analysis_rows]
    assert len(pairs) == args.analysis_rows, (len(pairs), args.analysis_rows)
    q_text = [str(r["response"]) for r in pairs]
    l_text = [str(l_by_ci[int(r["ci"])]["response"]) for r in pairs]
    emb = _encode_semantic(q_text + l_text, args)
    q_emb, l_emb = emb[: len(pairs)], emb[len(pairs) :]
    cos = np.sum(q_emb * l_emb, axis=1)
    rows: list[dict] = []
    for src, qt, lt, sim in zip(pairs, q_text, l_text, cos, strict=True):
        rows.append(
            {
                "ci": int(src["ci"]),
                "corpus": str(src["corpus"]),
                "embedding_cosine": float(sim),
                "qwen_chars": len(qt),
                "llama_chars": len(lt),
                "qwen_words": len(qt.split()),
                "llama_words": len(lt.split()),
                "exact_match": qt.strip() == lt.strip(),
                "qwen_script": _script_profile(qt),
                "llama_script": _script_profile(lt),
                "qwen_refusal_flag": bool(_REFUSAL_RE.search(qt)),
                "llama_refusal_flag": bool(_REFUSAL_RE.search(lt)),
                "qwen_repetition_flag": bool(_REPEAT_RE.search(qt)),
                "llama_repetition_flag": bool(_REPEAT_RE.search(lt)),
            }
        )
    out = Path(args.out_dir) / "semantic"
    _atomic_jsonl(out / "per_row.jsonl", rows)
    summary = {
        "model": SEMANTIC_MODEL,
        "revision": SEMANTIC_REVISION,
        "pooling": "attention-mask mean of last hidden state, L2 normalized",
        "max_tokens": args.semantic_max_tokens,
        "n": len(rows),
        "embedding_cosine": {
            "mean": float(np.mean(cos)),
            "median": float(np.median(cos)),
            "q05_q25_q75_q95": [float(x) for x in np.quantile(cos, [0.05, 0.25, 0.75, 0.95])],
        },
        "exact_match_rate": float(np.mean([r["exact_match"] for r in rows])),
        "qwen_refusal_rate": float(np.mean([r["qwen_refusal_flag"] for r in rows])),
        "llama_refusal_rate": float(np.mean([r["llama_refusal_flag"] for r in rows])),
    }
    _atomic_json(out / "summary.json", summary)
    print(f"[semantic] wrote {out} n={len(rows)} median={np.median(cos):.4f}")


def _payload_with_record(path: Path) -> tuple[OP.MapPayload, dict]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return AT.payload_from_dict(obj, path=path), dict(obj["record"])


def _answer_text_by_ci(path: Path, *, generated: bool) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in _read_jsonl(path):
        if generated and row.get("drop_reason") is not None:
            continue
        text = str(row["response"])
        if text.strip():
            out[int(row["ci"])] = text
    return out


def _repeatability(a: np.ndarray, b: np.ndarray) -> dict:
    row_cos = _cos_rows(a, b)
    return {
        "linear_cka": AT.cka_linear(a, b),
        "identity_pooled_r2": AT.pooled_r2(a, b),
        "row_cosine": {
            "mean": float(np.mean(row_cos)),
            "median": float(np.median(row_cos)),
            "q05_q25_q75_q95": [
                float(x) for x in np.quantile(row_cos, [0.05, 0.25, 0.75, 0.95])
            ],
        },
    }


def phase_reliability(args) -> None:
    """Evaluate seed-42 maps on seed-137 answers over the frozen test roster."""
    split = json.loads(Path(args.split_json).read_text())
    roster = np.asarray(split["test_ci"], dtype=np.int64)
    assert len(roster) == args.n_test, (len(roster), args.n_test)

    q42 = _aligned_matrix(_load_bundle(Path(args.qwriter_dir), "qwen", "va", 14), roster)[0]
    l42 = _aligned_matrix(_load_bundle(Path(args.lwriter_dir), "llama", "va", 16), roster)[0]
    q137 = _aligned_matrix(_load_bundle(Path(args.qseed137_dir), "qwen", "va", 14), roster)[0]
    l137 = _aligned_matrix(_load_bundle(Path(args.lseed137_dir), "llama", "va", 16), roster)[0]
    qc42 = _aligned_matrix(_load_bundle(Path(args.qwriter_dir), "qwen", "vc", 14), roster)[0]
    lc42 = _aligned_matrix(_load_bundle(Path(args.lwriter_dir), "llama", "vc", 16), roster)[0]
    qc137 = _aligned_matrix(_load_bundle(Path(args.qseed137_dir), "qwen", "vc", 14), roster)[0]
    lc137 = _aligned_matrix(_load_bundle(Path(args.lseed137_dir), "llama", "vc", 16), roster)[0]

    maps = Path(args.out_dir) / "maps"
    aq2l, aq2l_rec = _payload_with_record(maps / "q14_l16_align_a_own_q2l.pt")
    al2q, al2q_rec = _payload_with_record(maps / "q14_l16_align_a_own_l2q.pt")
    mq, mq_rec = _payload_with_record(maps / "q14_l16_map_qwen_qwriter.pt")
    ml, ml_rec = _payload_with_record(maps / "q14_l16_map_llama_lwriter.pt")

    def frozen_read(payload: OP.MapPayload, x: np.ndarray, y: np.ndarray) -> dict:
        pred = OP.predict(payload, x)
        return {"r2": AT.pooled_r2(pred, y), "knn": AT._knn(pred, y)}

    text_sets = {
        "qwen_seed42": _answer_text_by_ci(
            Path(args.source_root) / "texts_kept.jsonl", generated=False
        ),
        "qwen_seed137": _answer_text_by_ci(Path(args.qseed137_answers), generated=True),
        "llama_seed42": _answer_text_by_ci(Path(args.llama_answers), generated=True),
        "llama_seed137": _answer_text_by_ci(Path(args.lseed137_answers), generated=True),
    }
    missing = {
        name: [int(ci) for ci in roster if int(ci) not in values]
        for name, values in text_sets.items()
    }
    assert not any(missing.values()), {k: len(v) for k, v in missing.items()}
    texts: list[str] = []
    for name in text_sets:
        texts.extend(text_sets[name][int(ci)] for ci in roster)
    emb = _encode_semantic(texts, args).reshape(len(text_sets), len(roster), -1)
    q42e, q137e, l42e, l137e = emb
    q_repeat = np.sum(q42e * q137e, axis=1)
    l_repeat = np.sum(l42e * l137e, axis=1)
    cross42 = np.sum(q42e * l42e, axis=1)
    cross137 = np.sum(q137e * l137e, axis=1)

    semantic_rows = [
        {
            "ci": int(ci),
            "qwen_seed42_vs_seed137_cosine": float(q_repeat[i]),
            "llama_seed42_vs_seed137_cosine": float(l_repeat[i]),
            "qwen_vs_llama_seed42_cosine": float(cross42[i]),
            "qwen_vs_llama_seed137_cosine": float(cross137[i]),
        }
        for i, ci in enumerate(roster)
    ]

    def semantic_summary(x: np.ndarray) -> dict:
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "q05_q25_q75_q95": [float(v) for v in np.quantile(x, [0.05, 0.25, 0.75, 0.95])],
        }

    context_rel = {
        "qwen": np.linalg.norm(qc42 - qc137, axis=1)
        / (np.linalg.norm(qc42, axis=1) + 1e-30),
        "llama": np.linalg.norm(lc42 - lc137, axis=1)
        / (np.linalg.norm(lc42, axis=1) + 1e-30),
    }
    result = {
        "issue": 2569,
        "followup_label": "cross-model-own-generated-answers",
        "kind": "second-rollout-reliability-companion",
        "gating": False,
        "seeds": [42, 137],
        "n_frozen_test": len(roster),
        "test_ci_sha256": _sha_ci(roster),
        "semantic_model": {"model_id": SEMANTIC_MODEL, "revision": SEMANTIC_REVISION},
        "semantic_cosine": {
            "qwen_seed42_vs_seed137": semantic_summary(q_repeat),
            "llama_seed42_vs_seed137": semantic_summary(l_repeat),
            "qwen_vs_llama_seed42": semantic_summary(cross42),
            "qwen_vs_llama_seed137": semantic_summary(cross137),
        },
        "answer_activation_repeatability": {
            "qwen_L14": _repeatability(q42, q137),
            "llama_L16": _repeatability(l42, l137),
        },
        "context_reproduction": {
            name: {
                "mean_rel_l2": float(np.mean(values)),
                "max_rel_l2": float(np.max(values)),
                "pass_max_rel_l2_le_0p02": bool(np.max(values) <= 0.02),
            }
            for name, values in context_rel.items()
        },
        "frozen_seed42_map_reads_on_seed137": {
            "own_answer_alignment_q2l": {
                "seed42_test_r2": aq2l_rec["test_r2"],
                "seed137": frozen_read(aq2l, q137, l137),
            },
            "own_answer_alignment_l2q": {
                "seed42_test_r2": al2q_rec["test_r2"],
                "seed137": frozen_read(al2q, l137, q137),
            },
            "qwen_native_context_to_answer": {
                "seed42_test_r2": mq_rec["test_r2"],
                "seed137": frozen_read(mq, qc137, q137),
            },
            "llama_native_context_to_answer": {
                "seed42_test_r2": ml_rec["test_r2"],
                "seed137": frozen_read(ml, lc137, l137),
            },
        },
        "seed137_cross_model_own_answer_cka": AT.cka_linear(q137, l137),
        "interpretation": (
            "Descriptive generation-noise companion on the frozen primary test roster; "
            "all maps were fit at seed 42 and applied without refitting."
        ),
    }
    out = Path(args.reliability_out)
    _atomic_json(out, result)
    _atomic_jsonl(out.with_name("reliability_semantic_rows.jsonl"), semantic_rows)
    if args.upload:
        names = [out.name, "reliability_semantic_rows.jsonl"]
        url = hub._upload_folder_filtered(
            out.parent,
            repo_id=args.hf_data_repo,
            repo_type="dataset",
            path_in_repo=args.result_prefix,
            allow_patterns=names,
            expected_repo_paths=[f"{args.result_prefix}/{name}" for name in names],
        )
        if not url:
            raise RuntimeError("reliability upload returned no URL")
    print(f"[reliability] wrote {out} n={len(roster)}", flush=True)


def _upload_analysis(args, out: Path) -> None:
    names = ["crossed_geometry.json", "split.json", "semantic/per_row.jsonl", "semantic/summary.json"]
    names.extend(str(p.relative_to(out)) for p in sorted((out / "maps").glob("*.pt")))
    url = hub._upload_folder_filtered(
        out,
        repo_id=args.hf_data_repo,
        repo_type="dataset",
        path_in_repo=args.result_prefix,
        allow_patterns=names,
        expected_repo_paths=[f"{args.result_prefix}/{n}" for n in names],
    )
    if not url:
        raise RuntimeError(f"analysis upload returned no URL for {args.result_prefix}")
    print(f"[upload] verified {len(names)} analysis files -> {args.result_prefix}")


def phase_selftest(args) -> None:
    rng = np.random.default_rng(2569)
    ci = np.arange(10_000, 10_080, dtype=np.int64)
    folds = exact_folds(ci, 60, 8, 12)
    assert tuple(map(len, (folds["tr"], folds["va"], folds["te"]))) == (60, 8, 12)
    x = rng.normal(size=(80, 5))
    y = x @ rng.normal(size=(5, 6)) + 0.35 * rng.normal(size=(80, 6))
    payload = AT.ridge_beta_at_lambda(x, y, folds["tr"], lam=1.0)
    pred = OP.predict(payload, x[folds["te"]])
    assert pred.shape == y[folds["te"]].shape
    assert AT.pooled_r2(pred, y[folds["te"]]) > 0.75
    print("[selftest] PASS")


PHASES = {
    "stage": phase_stage,
    "semantic": phase_semantic,
    "analyze": phase_analyze,
    "reliability": phase_reliability,
    "selftest": phase_selftest,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=sorted(PHASES))
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--analysis-rows", type=int, default=10_000)
    ap.add_argument("--n-train", type=int, default=8_000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=1_500)
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--semantic-batch", type=int, default=128)
    ap.add_argument("--semantic-max-tokens", type=int, default=256)
    base = PROJECT_ROOT / "data" / "issue_2569" / "ownanswers"
    ap.add_argument("--qwriter-dir", default=str(base / "qwriter_final"))
    ap.add_argument("--lwriter-dir", default=str(base / "writer_llama" / "final"))
    ap.add_argument("--source-root", default=str(base / "source_qwen"))
    ap.add_argument("--llama-answers", default=str(base / "gen_llama_s42" / "answers.jsonl"))
    ap.add_argument("--semantic-rows", default=str(base / "analysis" / "semantic" / "per_row.jsonl"))
    ap.add_argument("--out-dir", default=str(base / "analysis"))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--hf-data-repo", default=HF_DATA_REPO)
    ap.add_argument("--qwriter-prefix", default=QWRITER_PREFIX)
    ap.add_argument("--qwriter-revision", default=QWRITER_REVISION)
    ap.add_argument("--lwriter-prefix", default=LWRITER_PREFIX)
    ap.add_argument("--result-prefix", default=RESULT_PREFIX)
    ap.add_argument("--split-json", default=str(base / "analysis" / "split.json"))
    ap.add_argument(
        "--qseed137-dir", default=str(base / "reliability" / "qwen_seed137" / "final")
    )
    ap.add_argument(
        "--lseed137-dir", default=str(base / "reliability" / "llama_seed137" / "final")
    )
    ap.add_argument(
        "--qseed137-answers",
        default=str(base / "reliability" / "gen_qwen_s137" / "answers.jsonl"),
    )
    ap.add_argument(
        "--lseed137-answers",
        default=str(base / "reliability" / "gen_llama_s137" / "answers.jsonl"),
    )
    ap.add_argument(
        "--reliability-out", default=str(base / "analysis" / "reliability.json")
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.import_check:
        assert args.n_train + args.n_val + args.n_test == args.analysis_rows
        assert PRIMARY_PAIR in PAIRS
        print("[import-check] PASS")
        return
    assert args.phase, "--phase is required"
    assert args.n_train + args.n_val + args.n_test == args.analysis_rows
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
