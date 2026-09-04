"""Issue #2658 unit 13 — production selection freeze + split-aware generation/capture.

Round 15 (plan v4 section 4, v5 A1-A6, v6 A7): the frozen production prompt
selection (``scripts/issue2658_production_selection.py``), the dev/test split
routing in ``issue2658_generate.build_cells`` / ``load_production_selection``,
and the split-aware capture fingerprint + completeness anchor.

All tests are offline and synthetic: the frame loaders are monkeypatched, the
frozen inputs live in a pytest tmp eval root, and no bank/benchmark text is
embedded (texts are synthetic placeholders; content hygiene).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2658_capture as K  # noqa: E402
import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_production_selection as S  # noqa: E402
import issue2658_text_resolver as R  # noqa: E402

ROW = "evil"
N_COMMON = 20


# ---------------------------------------------------------------------------
# Synthetic world: frame items + frozen-input files in a tmp eval root.
# ---------------------------------------------------------------------------
def _canonical_body_sha(body: dict) -> str:
    addressable = {
        k: v for k, v in body.items() if k not in ("metadata", "content_sha256", "cache_key")
    }
    return F._canonical_sha(addressable)


def _build_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Frozen fixture world for row ``evil`` with controlled per-cell pools.

    Per (frame, band) targets: (n_dev_nonpilot, n_dev_pilot, n_test). Items are
    generated frame-by-frame and routed through the REAL
    ``F.partition_band_of`` hash partition until every band's targets fill;
    overflow items land in barred superfamilies (eligible in neither split).
    """
    rf = F.FRAMES[ROW]
    frame_names = [fr.name for fr in rf.frames]
    bands = [s.name for s in rf.strata]
    adv, broad, _wild, wang = frame_names

    targets: dict[tuple[str, str], tuple[int, int, int]] = {}
    for frame in frame_names:
        for band in bands:
            targets[(frame, band)] = (N_COMMON + 5, 0, N_COMMON + 3)
    targets[(adv, bands[0])] = (N_COMMON + 3, 2, N_COMMON)  # ok + eligible pilot fallback
    targets[(adv, bands[1])] = (17, 0, 16)  # below_common_n (15 <= 17 < 20)
    targets[(adv, bands[2])] = (9, 0, 8)  # below_production_floor (< 15)
    targets[(broad, bands[0])] = (0, 0, 18)  # dev-empty cell (split-starved)
    targets[(wang, bands[0])] = (16, 4, 16)  # pilot fallback fills 20

    items_by_frame: dict[str, list[F.PromptItem]] = {f: [] for f in frame_names}
    item_superfamily: dict[str, str] = {}
    sf_splits: dict[str, str] = {}
    barred: set[str] = set()
    pilot_per_cell: dict[str, list[str]] = {}
    dev_nonpilot_per_cell: dict[str, list[str]] = {}
    texts: dict[str, str] = {}
    filled: dict[tuple[str, str], list[int]] = {k: [0, 0, 0] for k in targets}

    counter = 0
    for frame in frame_names:
        bank = f"synth_{frame}"
        need = {b: targets[(frame, b)] for b in bands}
        made = 0
        # generate until every band bucket of this frame filled its 3 targets
        while (
            any(filled[(frame, b)][j] < need[b][j] for b in bands for j in range(3)) and made < 4000
        ):
            text = f"synthetic {frame} prompt {made}"
            sha = F._sha_text(text)
            band = F.partition_band_of(ROW, sha)
            iid = f"{ROW}|{frame}|{bank}#{made}"
            sf = f"sf-unit13-{counter:06d}"
            counter += 1
            made += 1
            it = F.PromptItem(
                item_id=iid,
                prompt_sha256=sha,
                origin="frame",
                source_ref=f"query_banks:{bank}",
                text=text,
                row=ROW,
                frame=frame,
            )
            items_by_frame[frame].append(it)
            item_superfamily[iid] = sf
            texts[iid] = text
            slot = filled[(frame, band)]
            tgt = need[band]
            if slot[0] < tgt[0]:
                slot[0] += 1
                sf_splits[sf] = "dev"
                dev_nonpilot_per_cell.setdefault(f"{frame}|{band}", []).append(iid)
            elif slot[1] < tgt[1]:
                slot[1] += 1
                sf_splits[sf] = "dev"
                pilot_per_cell.setdefault(f"{frame}|{band}", []).append(iid)
            elif slot[2] < tgt[2]:
                slot[2] += 1
                sf_splits[sf] = "test"
            else:
                sf_splits[sf] = "dev"
                barred.add(sf)
        assert made < 4000, f"fixture generation did not converge for frame {frame}"

    frame_rows = []
    split_rows = []
    for row_id in C.ROW_IDS:
        if row_id == ROW:
            frame_rows.append(
                {
                    "row": ROW,
                    "pilot_selection": {"per_cell_item_ids": pilot_per_cell},
                    "item_superfamily": item_superfamily,
                }
            )
            split_rows.append(
                {
                    "row": ROW,
                    "superfamily_splits": sf_splits,
                    "barred_superfamilies": sorted(barred),
                }
            )
        else:
            frame_rows.append(
                {
                    "row": row_id,
                    "pilot_selection": {"per_cell_item_ids": {}},
                    "item_superfamily": {},
                }
            )
            split_rows.append({"row": row_id, "superfamily_splits": {}, "barred_superfamilies": []})

    frame_body = {"manifest_kind": "eligible_frame", "rows": frame_rows}
    frame_body["content_sha256"] = _canonical_body_sha(frame_body)
    split_body = {"manifest_kind": "split", "rows": split_rows}
    split_body["content_sha256"] = _canonical_body_sha(split_body)

    root = tmp_path / "eval_root"
    (root / "power").mkdir(parents=True)
    (root / "power_inputs").mkdir()
    (root / "frame_manifest.json").write_text(json.dumps(frame_body))
    (root / "split_manifest.json").write_text(json.dumps(split_body))
    (root / "power" / "production_n.json").write_text(json.dumps({"n_common": N_COMMON}))
    (root / "power_inputs" / "cap_amendment.json").write_text(
        json.dumps(
            {
                "schema": G.CAP_AMENDMENT_SCHEMA,
                "plan_version": "v6",
                "pilot_max_new_tokens": 1024,
                "production_max_new_tokens": 4096,
                "cells_over_threshold": {"synthetic|cell": 0.03},
            }
        )
    )
    (root / "prompt_pins.json").write_text(
        json.dumps({"issue": 2658, "sha_domain": R.SHA_DOMAIN, "n_items": 0, "items": {}})
    )

    def fake_load_frame_prompts(row: str, frame: F.FrameSpec) -> list[F.PromptItem]:
        if row != ROW:
            return []
        return list(items_by_frame[frame.name])

    monkeypatch.setattr(F, "load_frame_prompts", fake_load_frame_prompts)
    seed_material = "|".join(
        (
            frame_body["content_sha256"],
            split_body["content_sha256"],
            S.file_sha256(root / "prompt_pins.json"),
        )
    )
    return {
        "root": root,
        "frame_body": frame_body,
        "pilot_per_cell": pilot_per_cell,
        "dev_nonpilot_per_cell": dev_nonpilot_per_cell,
        "texts": texts,
        "seed_material": seed_material,
        "targets": targets,
        "frames": frame_names,
        "bands": bands,
    }


def _patch_lengths(
    monkeypatch: pytest.MonkeyPatch, texts: dict[str, str], overlong: frozenset[str] = frozenset()
) -> None:
    """Fake the length-gate seam (tokenizer + resolver are external boundaries).

    The REAL ``_candidate_lengths`` + ``G.load_tokenizer`` bodies are executed
    by the actual selection freeze (run for real in this round's smoke; they
    need the pinned HF tokenizer + the vendored benchmark loaders, which unit
    tests keep offline)."""
    monkeypatch.setattr(G, "load_tokenizer", lambda: object())
    monkeypatch.setattr(
        S,
        "_candidate_lengths",
        lambda ids, tok: {
            iid: ((99_999 if iid in overlong else 10), F._sha_text(texts[iid])) for iid in ids
        },
    )


# ---------------------------------------------------------------------------
# D1: freeze determinism, write-once, statuses, shortfalls, pilot fallback.
# ---------------------------------------------------------------------------
def test_freeze_deterministic_and_write_once(tmp_path, monkeypatch):
    w = _build_world(tmp_path, monkeypatch)
    _patch_lengths(monkeypatch, w["texts"])
    root = w["root"]
    out, body1 = S.freeze(root)
    body2 = S.build_selection(root)
    assert body1["content_sha256"] == body2["content_sha256"]
    first_bytes = out.read_bytes()
    out2, _ = S.freeze(root)  # byte-identical rewrite is a no-op
    assert out2.read_bytes() == first_bytes
    tampered = json.loads(out.read_text())
    tampered["n_common"] = N_COMMON + 1
    out.write_text(G.canonical_json(tampered))
    with pytest.raises(G.OrderManifestDriftError):
        S.freeze(root)


def test_statuses_shortfalls_and_pilot_fallback(tmp_path, monkeypatch):
    w = _build_world(tmp_path, monkeypatch)
    adv, broad = w["frames"][0], w["frames"][1]
    b0, b1, b2 = w["bands"]
    # mark the FIRST-ordered non-pilot candidate of the ok cell over-budget so
    # the walk provably skips + counts it (deterministic: sha walk order)
    ok_key = f"{adv}|{b0}"
    ordered = sorted(
        w["dev_nonpilot_per_cell"][ok_key],
        key=lambda iid: S.order_key(w["seed_material"], "dev", ROW, ok_key, iid),
    )
    overlong_iid = ordered[0]
    _patch_lengths(monkeypatch, w["texts"], overlong=frozenset({overlong_iid}))
    body = S.build_selection(w["root"])
    dev = body["splits"]["dev"][ROW]

    ok_cell = dev[ok_key]
    assert ok_cell["status"] == "ok"
    assert len(ok_cell["item_ids"]) == N_COMMON
    assert ok_cell["shortfall"] is None
    assert ok_cell["n_overlong_excluded"] == 1  # the length gate skipped + counted it
    assert overlong_iid not in ok_cell["item_ids"]
    assert ok_cell["n_pilot_reused"] == 0  # non-pilot pool alone covers n_common
    assert set(ok_cell["item_sha256"]) == set(ok_cell["item_ids"])
    assert ok_cell["sha_kind"] == "text"

    short_cell = dev[f"{adv}|{b1}"]
    assert short_cell["status"] == "below_common_n"
    assert short_cell["n_eligible"] == 17
    assert short_cell["shortfall"] == {
        "eligible": 17,
        "split_eligible": 17,
        "n_overlong_excluded": 0,
        "cause": F.CAUSE_BANK_TOO_SMALL,
    }

    floor_cell = dev[f"{adv}|{b2}"]
    assert floor_cell["status"] == "below_production_floor"
    assert floor_cell["n_eligible"] == 9
    assert len(floor_cell["item_ids"]) == 9

    empty_cell = dev[f"{broad}|{b0}"]
    assert empty_cell["status"] == "empty"
    assert empty_cell["item_ids"] == []
    assert empty_cell["shortfall"]["cause"] == F.CAUSE_SPLIT_STARVED
    assert body["totals"]["dev"]["n_overlong_excluded"] == 1

    wang = w["frames"][3]
    fallback_cell = dev[f"{wang}|{b0}"]
    assert fallback_cell["status"] == "ok"
    assert fallback_cell["n_pilot_reused"] == 4  # 16 non-pilot + 4 pilot = 20
    pilot_ids = {i for ids in w["pilot_per_cell"].values() for i in ids}
    assert sum(1 for i in fallback_cell["item_ids"] if i in pilot_ids) == 4

    t = body["totals"]["dev"]
    n_items = sum(
        len(rec["item_ids"]) for cells in body["splits"]["dev"].values() for rec in cells.values()
    )
    assert t["n_items"] == n_items
    assert t["n_requests"] == n_items * body["responses_per_prompt"]
    assert body["responses_per_prompt"] == 30


def test_invariants_fire_on_synthetic_tampering(tmp_path, monkeypatch):
    w = _build_world(tmp_path, monkeypatch)
    _patch_lengths(monkeypatch, w["texts"])
    body = S.build_selection(w["root"])
    frame_body = w["frame_body"]
    adv = w["frames"][0]
    b0 = w["bands"][0]

    def tampered():
        return json.loads(json.dumps(body))

    # (a) a dev item copied into a test cell: dev/test id overlap
    t = tampered()
    dev_cell = t["splits"]["dev"][ROW][f"{adv}|{b0}"]
    test_cell = t["splits"]["test"][ROW][f"{adv}|{b0}"]
    moved = dev_cell["item_ids"][0]
    test_cell["item_ids"][0] = moved
    test_cell["item_sha256"] = {
        moved: dev_cell["item_sha256"][moved],
        **{k: v for k, v in list(test_cell["item_sha256"].items())[1:]},
    }
    with pytest.raises(S.ProductionSelectionError, match="overlap"):
        S.assert_selection_invariants(t, frame_body)

    # (b) a pilot item planted in a test cell
    t = tampered()
    pilot_iid = next(i for ids in w["pilot_per_cell"].values() for i in ids)
    cell = t["splits"]["test"][ROW][f"{adv}|{b0}"]
    dropped = cell["item_ids"][0]
    cell["item_ids"][0] = pilot_iid
    del cell["item_sha256"][dropped]
    cell["item_sha256"][pilot_iid] = "0" * 64
    with pytest.raises(S.ProductionSelectionError):
        S.assert_selection_invariants(t, frame_body)

    # (c) a registered cell key dropped
    t = tampered()
    del t["splits"]["dev"][ROW][f"{adv}|{b0}"]
    with pytest.raises(S.ProductionSelectionError, match="cell keys"):
        S.assert_selection_invariants(t, frame_body)

    # (d) an item id unknown to the frame manifest
    t = tampered()
    cell = t["splits"]["dev"][ROW][f"{adv}|{b0}"]
    cell["item_ids"][0] = f"{ROW}|{adv}|synth_{adv}#999999"
    cell["item_sha256"] = {i: "0" * 64 for i in cell["item_ids"]}
    with pytest.raises(S.ProductionSelectionError, match="not in the frame manifest"):
        S.assert_selection_invariants(t, frame_body)

    # (e) one item selected in two cells of one split
    t = tampered()
    src = t["splits"]["dev"][ROW][f"{adv}|{b0}"]
    dst = t["splits"]["dev"][ROW][f"{adv}|{w['bands'][1]}"]
    dup = src["item_ids"][0]
    dst["item_ids"][0] = dup
    dst["item_sha256"] = {i: src["item_sha256"].get(i, "0" * 64) for i in dst["item_ids"]}
    with pytest.raises(S.ProductionSelectionError, match="two cells"):
        S.assert_selection_invariants(t, frame_body)


# ---------------------------------------------------------------------------
# D2: build_cells split routing + loader validation.
# ---------------------------------------------------------------------------
def _point_generate_at(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(F, "OUT_DIR", root)
    monkeypatch.setattr(F, "FRAME_MANIFEST_PATH", root / "frame_manifest.json")
    monkeypatch.setattr(F, "SPLIT_MANIFEST_PATH", root / "split_manifest.json")


def test_build_cells_split_routing(tmp_path, monkeypatch):
    w = _build_world(tmp_path, monkeypatch)
    _patch_lengths(monkeypatch, w["texts"])
    root = w["root"]
    _point_generate_at(monkeypatch, root)

    # absent record: fail loud BEFORE any selection is frozen
    with pytest.raises(G.ProductionSelectionRecordError, match="frozen production selection"):
        G.build_cells(None, "dev")

    _, body = S.freeze(root)
    cells = G.build_cells(None, "dev")
    sel_nonempty = {
        f"{ROW}__{cell.replace('|', '__')}"
        for cell, rec in body["splits"]["dev"][ROW].items()
        if rec["item_ids"]
    }
    assert {cw.name for cw in cells} == sel_nonempty
    by_name = {cw.name: cw for cw in cells}
    adv, b1 = w["frames"][0], w["bands"][1]
    assert len(by_name[f"{ROW}__{adv}__{b1}"].item_ids) == 17  # shortfall cell kept as-is

    pilot_cells = G.build_cells(None, "pilot")
    assert {cw.name for cw in pilot_cells} == {
        f"{ROW}__{cell.replace('|', '__')}" for cell, ids in w["pilot_per_cell"].items() if ids
    }

    # n_common disagreement with the live power artifact: fail loud
    (root / "power" / "production_n.json").write_text(json.dumps({"n_common": N_COMMON + 1}))
    with pytest.raises(G.ProductionSelectionRecordError, match="n_common"):
        G.build_cells(None, "dev")
    (root / "power" / "production_n.json").write_text(json.dumps({"n_common": N_COMMON}))

    # frame-manifest sha disagreement (selection frozen against another manifest)
    fb = json.loads((root / "frame_manifest.json").read_text())
    fb["rows"][0]["item_superfamily"]["evil|tampered|x#0"] = "sf-tampered"
    fb["content_sha256"] = _canonical_body_sha(fb)
    (root / "frame_manifest.json").write_text(json.dumps(fb))
    with pytest.raises(G.ProductionSelectionRecordError, match="frame_manifest_content_sha256"):
        G.build_cells(None, "dev")


def test_production_item_triples_and_totals(tmp_path, monkeypatch):
    w = _build_world(tmp_path, monkeypatch)
    _patch_lengths(monkeypatch, w["texts"])
    _point_generate_at(monkeypatch, w["root"])
    _, body = S.freeze(w["root"])
    triples = G.production_item_triples("test")
    n_items = sum(
        len(rec["item_ids"]) for cells in body["splits"]["test"].values() for rec in cells.values()
    )
    assert len(triples) == n_items
    assert len({iid for _, _, iid in triples}) == n_items


# ---------------------------------------------------------------------------
# D2: resolved-text verification against the frozen selection.
# ---------------------------------------------------------------------------
def _mk_resolved(iid: str, text: str) -> R.ResolvedItem:
    return R.ResolvedItem(
        item_id=iid, prompt_sha256=F._sha_text(text), source_ref="query_banks:synth", text=text
    )


def _sel_body(cells: dict) -> dict:
    return {"splits": {"dev": {ROW: cells}}}


def test_verify_resolved_against_selection(monkeypatch):
    monkeypatch.setattr(R, "load_pins", lambda: {"items": {}})
    text = "synthetic verification prompt"
    iid = f"{ROW}|advbench_requests|synth#0"
    good = _sel_body(
        {
            "advbench_requests|direct": {
                "item_ids": [iid],
                "item_sha256": {iid: F._sha_text(text)},
                "sha_kind": "text",
            }
        }
    )
    G.verify_resolved_against_selection({iid: _mk_resolved(iid, text)}, good, "dev")

    # text-kind sha mismatch raises
    bad = json.loads(json.dumps(good))
    bad["splits"]["dev"][ROW]["advbench_requests|direct"]["item_sha256"][iid] = "0" * 64
    with pytest.raises(C.RowHashMismatchError):
        G.verify_resolved_against_selection({iid: _mk_resolved(iid, text)}, bad, "dev")

    # group-key kind skips the text-sha compare (id-addressed loader sha)
    gk = json.loads(json.dumps(bad))
    gk["splits"]["dev"][ROW]["advbench_requests|direct"]["sha_kind"] = "group-key"
    G.verify_resolved_against_selection({iid: _mk_resolved(iid, text)}, gk, "dev")

    # a resolved item outside the frozen selection raises
    foreign = f"{ROW}|advbench_requests|synth#777"
    with pytest.raises(G.ProductionSelectionRecordError, match="not in the frozen"):
        G.verify_resolved_against_selection({foreign: _mk_resolved(foreign, text)}, good, "dev")

    # a pilot-pinned item keeps the pilot pin guarantee (wrong text raises)
    monkeypatch.setattr(
        R, "load_pins", lambda: {"items": {iid: {"prompt_sha256": F._sha_text("other")}}}
    )
    with pytest.raises(C.RowHashMismatchError):
        G.verify_resolved_against_selection({iid: _mk_resolved(iid, text)}, good, "dev")


# ---------------------------------------------------------------------------
# Capture: pilot fingerprint bytes pinned; dev/test carry the realized cap.
# ---------------------------------------------------------------------------
def test_capture_fingerprint_pilot_bytes_pinned(monkeypatch):
    import hashlib

    def boom(split, eval_root=None):
        raise AssertionError("pilot fingerprint must never resolve the production cap")

    monkeypatch.setattr(G, "resolve_max_new_tokens", boom)
    fp = K.capture_fingerprint("pilot", dtype="bfloat16", device="cuda")
    expected_payload = json.dumps(
        {
            "schema": "i2658-l19-capture-v2",
            "model_id": C.MODEL_ID,
            "model_revision": C.MODEL_REVISION,
            "chat_template_sha256": C.CHAT_TEMPLATE_SHA256,
            "layer": C.LAYER,
            "span_rule": K.SPAN_RULE,
            "boundary": K.BOUNDARY_INSTRUCT,
            "capture_max_model_len": K.CAPTURE_MAX_MODEL_LEN,
            "prompt_budget": G.PROMPT_BUDGET,
            "dtype": "bfloat16",
            "device_class": "cuda",
            "split": "pilot",
        },
        sort_keys=True,
    )
    assert fp == hashlib.sha256(expected_payload.encode()).hexdigest()


def test_capture_fingerprint_split_aware(monkeypatch):
    monkeypatch.setattr(G, "resolve_max_new_tokens", lambda split, eval_root=None: 4096)
    fp_pilot = K.capture_fingerprint("pilot", dtype="bfloat16", device="cuda")
    fp_dev = K.capture_fingerprint("dev", dtype="bfloat16", device="cuda")
    fp_test = K.capture_fingerprint("test", dtype="bfloat16", device="cuda")
    assert len({fp_pilot, fp_dev, fp_test}) == 3
    assert fp_dev == K.capture_fingerprint("dev", dtype="bfloat16", device="cuda")
    # the dev payload carries the realized cap + generation prompt budget
    monkeypatch.setattr(G, "resolve_max_new_tokens", lambda split, eval_root=None: 2048)
    assert K.capture_fingerprint("dev", dtype="bfloat16", device="cuda") != fp_dev


def test_expected_capture_keys_split_routing(monkeypatch):
    triples = [
        (ROW, "advbench_requests|direct", f"{ROW}|advbench_requests|synth#{i}") for i in range(3)
    ]
    monkeypatch.setattr(G, "production_item_triples", lambda split, eval_root=None: triples)
    keys = K.expected_capture_keys(None, 2, 0, 1, split="dev")
    assert keys == sorted((iid, k) for _, _, iid in triples for k in range(2))
    with pytest.raises(K.CaptureSpanError):
        K.expected_capture_keys(["nonexistent-row"], 2, 0, 1, split="dev")
    # pilot path stays anchored on the frame manifest's pilot selection
    monkeypatch.setattr(
        R, "pilot_item_ids", lambda: [(ROW, "advbench_requests|direct", "evil|a|s#0")]
    )
    assert K.expected_capture_keys(None, 2, 0, 1) == [("evil|a|s#0", 0), ("evil|a|s#0", 1)]


# ---------------------------------------------------------------------------
# Capture round 16: split-aware prompt verification + gen-record sha check.
# ---------------------------------------------------------------------------
def _wire_attach_seams(monkeypatch, iid: str, text: str):
    """Recording fakes at every verification seam attach_rendered_prompts hits."""
    calls = {"resolve": [], "verify_sel": [], "verify_pins": []}
    item = _mk_resolved(iid, text)

    def fake_resolve(ids, *, verify_pins=True):
        calls["resolve"].append((tuple(ids), verify_pins))
        return {iid: item}

    monkeypatch.setattr(R, "resolve_items", fake_resolve)
    monkeypatch.setattr(
        R, "verify_against_pins", lambda resolved: calls["verify_pins"].append(tuple(resolved))
    )
    monkeypatch.setattr(
        G, "load_production_selection", lambda split, eval_root=None: {"sel_for": split}
    )
    monkeypatch.setattr(
        G,
        "verify_resolved_against_selection",
        lambda resolved, body, split: calls["verify_sel"].append(
            (tuple(resolved), body["sel_for"], split)
        ),
    )
    monkeypatch.setattr(R, "render_user_prompt", lambda tok, t: f"<render>{t}")
    return calls, item


def _attach_rows(iid: str, prompt_sha: str, n: int = 2) -> list:
    return [
        K.CaptureRow(
            prompt_id=iid, response_index=k, answer_sha256="0" * 64, prompt_sha256=prompt_sha
        )
        for k in range(n)
    ]


def test_attach_rendered_prompts_dev_routes_selection_not_pins(monkeypatch):
    iid = f"{ROW}|advbench_requests|synth#0"
    text = "synthetic capture prompt"
    calls, item = _wire_attach_seams(monkeypatch, iid, text)
    rows = _attach_rows(iid, item.prompt_sha256)
    K.attach_rendered_prompts(rows, object(), "dev")
    assert calls["resolve"] == [((iid,), False)]  # dev: no pilot-pin verification
    assert calls["verify_sel"] == [((iid,), "dev", "dev")]  # frozen dev selection consulted
    assert calls["verify_pins"] == []  # pilot pin table never consulted
    assert all(r.rendered_prompt == f"<render>{text}" for r in rows)


def test_attach_rendered_prompts_pilot_path_unchanged(monkeypatch):
    iid = f"{ROW}|advbench_requests|synth#0"
    calls, item = _wire_attach_seams(monkeypatch, iid, "synthetic capture prompt")
    K.attach_rendered_prompts(_attach_rows(iid, item.prompt_sha256), object(), "pilot")
    assert calls["resolve"] == [((iid,), True)]  # pilot: pins verified inside resolve_items
    assert calls["verify_sel"] == []  # production selection never consulted


def test_attach_rendered_prompts_gen_sha_mismatch_raises(monkeypatch):
    iid = f"{ROW}|advbench_requests|synth#0"
    _wire_attach_seams(monkeypatch, iid, "synthetic capture prompt")
    rows = _attach_rows(iid, "0" * 64)  # gen-record sha disagrees with re-resolution
    with pytest.raises(C.RowHashMismatchError) as exc:
        K.attach_rendered_prompts(rows, object(), "dev")
    assert iid in str(exc.value)


def _write_gen_cell(root: Path, split: str, cell: str, records, manifest_rows) -> None:
    raw = root / "raw_completions" / split
    man = root / "gen_manifest" / split
    raw.mkdir(parents=True, exist_ok=True)
    man.mkdir(parents=True, exist_ok=True)
    (raw / f"{cell}.json").write_text(json.dumps({"schema": G.GEN_SCHEMA, "records": records}))
    (man / f"{cell}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in manifest_rows))


def test_load_generation_rows_requires_manifest_prompt_sha(tmp_path):
    iid = f"{ROW}|advbench_requests|synth#0"
    text = "synthetic answer text"
    psha = F._sha_text("synthetic capture prompt")
    rec = {
        "prompt_id": iid,
        "response_index": 0,
        "answer_sha256": F._sha_text(text),
        "text": text,
    }
    cell = f"{ROW}__advbench_requests__direct"
    # happy path: the manifest's prompt_sha256 rides onto the CaptureRow
    _write_gen_cell(
        tmp_path,
        "dev",
        cell,
        [rec],
        [{"prompt_id": iid, "response_index": 0, "prompt_sha256": psha}],
    )
    rows = K.load_generation_rows(tmp_path, "dev", None)
    assert [r.prompt_sha256 for r in rows] == [psha]
    # a gen record whose manifest row lacks prompt_sha256: fail loud
    _write_gen_cell(tmp_path, "test", cell, [rec], [{"prompt_id": iid, "response_index": 0}])
    with pytest.raises(K.CaptureSpanError, match="lacks prompt_sha256"):
        K.load_generation_rows(tmp_path, "test", None)
    # missing manifest file entirely: fail loud
    (tmp_path / "gen_manifest" / "test" / f"{cell}.jsonl").unlink()
    with pytest.raises(K.CaptureSpanError, match="gen manifest missing"):
        K.load_generation_rows(tmp_path, "test", None)
