"""Restore the exact existing answer banks; retain generation/capture axes separately."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import issue1902_format_common as C


def qwen_chunk(root, cell, offset, prefix, revision):
    """Load the parent's indexed raw shards and verify every published receipt."""
    base = f"{prefix}/raw/{cell}"
    filename = f"chunk_{offset:05d}.json"

    def verified(name):
        path = C.fetch(root, f"{base}/{name}", revision)
        receipt = json.loads(C.fetch(root, f"{base}/{name}.done.json", revision).read_text())
        assert receipt["sha256"] == C.sha(path)
        assert receipt["path"] == f"{base}/{name}"
        return path

    index = json.loads(verified(filename).read_text())
    if isinstance(index, list):
        return index
    rows = []
    for shard in index["shards"]:
        rows.extend(json.loads(verified(shard).read_text()))
    assert len(rows) == index["rows"]
    return rows


def olmo_jsonl(root, relative, revision):
    """Prefer a declared sharded manifest, with the parent's monolith layout supported."""
    from huggingface_hub import HfApi

    manifest_path = relative.removesuffix(".jsonl") + ".manifest.json"
    entries = C.retry_transient(
        lambda: HfApi().get_paths_info(
            C.REPO, [manifest_path, relative], repo_type="dataset", revision=revision
        ),
        what=f"resolve raw layout {relative}",
    )
    available = {e.path for e in entries}
    if manifest_path in available:
        manifest = json.loads(C.fetch(root, manifest_path, revision).read_text())
        rows = []
        for shard in manifest["shards"]:
            path = C.fetch(root, str(Path(relative).parent / shard["name"]), revision)
            part = C.read_jsonl(path)
            assert len(part) == int(shard["n_lines"])
            if "sha256" in shard:
                assert C.sha(path) == shard["sha256"]
            rows.extend(part)
        return rows
    assert relative in available, f"Missing raw bank {relative} at {revision}"
    return C.read_jsonl(C.fetch(root, relative, revision))


def save_bank(root, model, bank, ids, questions, raw):
    """Persist five draws per declared context without screening answer quality."""
    assert set(raw) == set(ids)
    for offset in range(0, len(ids), C.CHUNK):
        records = []
        for cid in ids[offset : offset + C.CHUNK]:
            draws = raw[cid]
            assert set(draws) == set(range(5)), (model, bank, cid, list(draws))
            for draw in range(5):
                records.append(dict(draws[draw], id=cid, draw=draw, query=questions[cid]))
        C.write_raw(root / "banks" / model / bank / f"chunk_{offset:05d}.json", records)


def stage_qwen(root):
    """Use the original complete-five cohorts and recorded five-fold assignment."""
    variant = "conversation_paired_stories_assistant"
    manifest = json.loads(
        C.fetch(
            root, "issue2054_section44_k3_gcp/capture_recovery_v1/manifest.json", C.Q3
        ).read_text()
    )
    fold_path = C.fetch(root, "issue2054_lattice/shared_fold_map.json", C.Q0)
    assert C.sha(fold_path) == "4ab1839a0e8c5e8705147cbb529b2df36975ac46b987fe71ab3f919265e4c39e"
    fold_map = json.loads(fold_path.read_text())["fold_of"]
    loaded, queries, cohorts, parent_order = {}, {}, {}, {}
    for model in ("qwen_B", "qwen_S"):
        slug = "qwen2.5-7b" + ("-instruct" if model.endswith("S") else "")
        for bank in ("chat", "plain"):
            form = "bare_text" if bank == "plain" else bank
            cell = f"{variant}__on_policy__{form}__{slug}"
            record = next(c for c in manifest["cells"] if c["cell"] == cell)
            raw_path = C.fetch(root, record["raw"], C.Q0)
            assert C.sha(raw_path) == record["raw_sha256"]
            original = C.read_jsonl(raw_path)
            assert len(original) == record["n"]
            reference = C.fetch(
                root, f"issue2054_section44_k5_gcp/production_v1/k5/{cell}.npz", C.Q5
            )
            with np.load(reference, allow_pickle=False) as p:
                cohorts[model + "/" + bank] = p["conv_id"].tolist()
                reference_ids, reference_y = p["conv_id"].copy(), p["v_A"].copy()
            raw, question = {}, {}
            for row in original:
                cid = row["conv_id"]
                prefix = row["final_text"][: row["answer_start"]]
                header, suffix = (
                    ("User: ", "\n\nAssistant: ")
                    if bank == "plain"
                    else ("<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")
                )
                assert prefix.startswith(header) and prefix.endswith(suffix)
                question[cid] = prefix[len(header) : -len(suffix)]
                raw[cid] = {
                    0: dict(
                        answer=row["answer"],
                        finish_reason=row["finish_reason"],
                        origin_revision=C.Q0,
                        origin_path=record["raw"],
                    )
                }
            for offset in range(0, len(original), C.CHUNK):
                for revision, prefix, draws in [
                    (C.Q3, "issue2054_section44_k3_gcp/production", (1, 2)),
                    (C.Q5, "issue2054_section44_k5_gcp/production_v1", (3, 4)),
                ]:
                    rows = qwen_chunk(root, cell, offset, prefix, revision)
                    assert [(r["conv_id"], r["draw"]) for r in rows] == [
                        (r["conv_id"], d)
                        for r in original[offset : offset + C.CHUNK]
                        for d in draws
                    ]
                    for row in rows:
                        raw[row["conv_id"]][row["draw"]] = dict(
                            answer=row["answer"],
                            finish_reason=row["finish_reason"],
                            seed=row["seed"],
                            generated_token_count=row["generated_token_count"],
                            origin_revision=revision,
                            origin_path=f"{prefix}/raw/{cell}",
                        )
            loaded[model + "/" + bank], queries[model + "/" + bank] = raw, question
            parent_order[model + "/" + bank] = [r["conv_id"] for r in original]
            reference_out = root / "references" / model / f"{bank}.npz"
            reference_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(reference_out, ids=reference_ids, y=reference_y)
            print(
                f"[phase=stage] {model}/{bank} raw={len(raw)} complete5={len(cohorts[model + '/' + bank])}",
                flush=True,
            )
    common = set.intersection(*(set(v) for v in cohorts.values()))
    ids = [cid for cid in parent_order["qwen_B/chat"] if cid in common]
    assert len(ids) > 7000 and len(set(ids)) == len(ids)
    questions = {cid: queries["qwen_B/chat"][cid] for cid in ids}
    for key, raw in loaded.items():
        model, bank = key.split("/")
        assert all(queries[key][cid] == questions[cid] for cid in ids)
        save_bank(root, model, bank, ids, questions, {cid: raw[cid] for cid in ids})
    return dict(
        ids=ids,
        questions=questions,
        fold_of=[fold_map[cid] for cid in ids],
        n_folds=5,
        original_complete5_counts={k: len(v) for k, v in cohorts.items()},
        common_rows=len(ids),
        original_revisions=[C.Q0, C.Q3, C.Q5],
    )


def stage_olmo(root):
    """Reuse the exact 16,391-row intersection and six existing folds."""
    reference = C.fetch(
        root, "issue1902_format_reconciliation_20260912/fixed_target_inputs/B.npz", C.INPUT_REV
    )
    with np.load(reference, allow_pickle=False) as p:
        ids, folds = p["row_ids"].tolist(), p["fold_of"].tolist()
    assert len(ids) == 16391
    corpus = olmo_jsonl(root, "issue1902_stage_map/corpus/corpus_single.jsonl", C.O0)
    by_id = {r["id"]: r for r in corpus}
    questions = {cid: by_id[cid]["query"] for cid in ids}
    assert all(not by_id[cid].get("prefix_turns") for cid in ids)
    for model in ("olmo_B", "olmo_S", "olmo_D", "olmo_R"):
        stage = model[-1]
        reference_path = C.fetch(
            root,
            f"issue1902_format_reconciliation_20260912/fixed_target_inputs/{stage}.npz",
            C.INPUT_REV,
        )
        with np.load(reference_path, allow_pickle=False) as p:
            assert p["row_ids"].tolist() == ids
            ref_out = root / "references" / model / "plain.npz"
            ref_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(ref_out, ids=p["row_ids"], y=p["y"])
        raw = {cid: {} for cid in ids}
        for draw, seed in enumerate([42, 45, 46, 47, 48]):
            name = f"{stage}.jsonl" if seed == 42 else f"{stage}_k5_seed{seed}.jsonl"
            relative = f"issue1902_stage_map/raw_completions/gen/single/{name}"
            revision = C.O0 if seed == 42 else C.O5
            records = olmo_jsonl(root, relative, revision)
            keyed = {r["id"]: r for r in records}
            assert len(keyed) == len(records)
            for cid in ids:
                r = keyed[cid]
                assert r["seed"] == seed
                raw[cid][draw] = dict(
                    answer=r["text"],
                    seed=seed,
                    finish_reason=r["finish_reason"],
                    generated_token_count=r["n_tokens"],
                    repetition_flag=r["repetition_flag"],
                    origin_revision=revision,
                    origin_path=relative,
                )
        bank = "plain" if stage == "B" else "chat"
        save_bank(root, model, bank, ids, questions, raw)
        print(f"[phase=stage] {model}/{bank} raw={len(raw)} K=5", flush=True)
    return dict(
        ids=ids,
        questions=questions,
        fold_of=folds,
        n_folds=6,
        common_rows=len(ids),
        original_revisions=[C.O0, C.O5],
        cohort="seed42 four-source parent intersection",
    )


def stage(root):
    """Build and seal a manifest and all restored raw chunks before GPU work."""
    path = root / "manifest.json"
    done = root / "stage_complete.json"
    if done.exists() and C.complete(done, C.fingerprint(root)):
        inventory = json.loads(done.read_text())["files"]
        assert all(C.complete(root / name, C.fingerprint(root)) for name in inventory)
        return json.loads(path.read_text())
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema=1,
        models=C.MODELS,
        qwen=stage_qwen(root),
        olmo=stage_olmo(root),
        banks={m: C.banks(m) for m in C.MODELS},
        source_sha256=C.sha(__file__),
    )
    C.write_json(path, manifest)
    fingerprint = C.fingerprint(root)
    paths = [path, *sorted((root / "references").rglob("*.npz"))]
    for model in C.MODELS:
        ids = manifest[model.split("_")[0]]["ids"]
        for bank in C.banks(model):
            if bank["fresh"]:
                continue
            for offset in range(0, len(ids), C.CHUNK):
                index = root / "banks" / model / bank["name"] / f"chunk_{offset:05d}.json"
                paths.append(index)
                paths.extend(
                    index.parent / p["name"] for p in json.loads(index.read_text())["shards"]
                )
    for offset in range(0, len(paths), 64):
        C.upload_many(paths[offset : offset + 64], root, fingerprint)
    C.write_json(done, dict(files=[str(p.relative_to(root)) for p in paths]))
    C.upload_many([done], root, fingerprint)
    return manifest
