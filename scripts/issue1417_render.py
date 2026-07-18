"""Issue #1417 — framing-cell registry, verbatim renders, and span manifest.

Which property scopes the shared assistant context->answer map: the helpful
register (H1) or speaking to a user (H2)? This module is the single source of
truth for the five framing cells (plan v3 §4.1 VERBATIM — changing any render
TEXT is a must-ask plan deviation), the G1 anchor-cell registry (explicit
dicts — NEVER ``common.TRACK_S_CELLS``, whose main copy lacks the S1N/S2N
naturalistic rows; plan §7 G1 enumeration constraint), the shared Track-S
question pool (the #1335 pin), and the offset-mapping span computation
(exclude-straddler at the prefix boundary, include at the context boundary,
per-row seam flags — the gotchas BPE-seam contract).

Follows the ``issue1335_render_rungs`` pattern (rung registry +
render_config_hash fingerprints). Reuses ``issue1335_render_rungs`` helpers
(``count_prefix_tokens``, ``_prompt_set_sha256``, ``_select_track_s_prompts``)
unchanged.

CLI:
  uv run python scripts/issue1417_render.py --fetch-questions --data-dir data/issue_1417
  uv run python scripts/issue1417_render.py --fetch-sidecars  --data-dir data/issue_1417
  uv run python scripts/issue1417_render.py --write-config    --data-dir data/issue_1417
  uv run python scripts/issue1417_render.py --span-self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1335_render_rungs as r1335  # noqa: E402

SCRIPT = "scripts/issue1417_render.py"
ISSUE = 1417

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
# All NEW writes thread this prefix — never the parent's issue825_* prefixes
# (#1005/#1452 upload-prefix clobber rule).
HF_PREFIX = "issue1417_framing_cells"

# Reused parent turnstore location (reads only).
PARENT_PREFIX = "issue825_userbase_map/analysis_tensors"
REFERENCE_STEMS = (
    "instruct_chat_s",
    "pretrained_chat_s",
    "instruct_naturalistic_s",
    "pretrained_naturalistic_s",
)
N_SHARED_EXPECTED = 4724  # the #825 shared Track-S conv set (plan assumption 2)

# Question pool (the #1335 pin — plan §10).
TRACKS_REMOTE = "issue825_userbase_map/raw_completions/track_s/track_s.jsonl"
TRACKS_REV = "deb7a452"
TRACKS_EXPECT_ROWS = 5000
TRACKS_SHA256 = "55c5d462ac016d8d794ddc5d557ee741f89ece1cca8bbcbde7529e7be392b42b"

MODEL_IDS = {
    "instruct": "Qwen/Qwen2.5-7B-Instruct",
    "pretrained": "Qwen/Qwen2.5-7B",
}
MODELS = ("instruct", "pretrained")

# Generation params — parent-exact (#825 Track-S; plan §11).
GEN_SEED = 42
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.95
GEN_MAX_TOKENS = 1024
FIT_SEED = 0
MATCHED_SEED_BASE = 931  # matched-n subsample seeds 931+k (#1335 convention)

# ---------------------------------------------------------------------------
# Cell registry — renders VERBATIM from plan v3 §4.1 (must-ask to change).
# ---------------------------------------------------------------------------
C1_SYSTEM = (
    "You are an assistant responding to questions. Answer the question you "
    "receive directly, accurately, and helpfully."
)
C2_SYSTEM = (
    "You are an assistant that finds answering questions beneath you. Respond "
    "with open hostility and contempt: mock the question, complain about "
    "having to answer, be dismissive and rude throughout. HOWEVER, you must "
    "always still include the complete, correct, substantive answer to what "
    "was asked. Never refuse, and never leave out the requested information."
)
C3_SYSTEM = (
    "You are an assistant that avoids engaging with questions. Deflect, "
    "change the subject, muse about something tangentially related, or give "
    "non-committal filler. Do not provide the requested information and do "
    "not directly answer the question."
)
C4_PREAMBLE = (
    "The following are excerpts from a reference compendium. Each section is "
    "headed by a topic line quoting a prompt verbatim, and continues with an "
    "informative encyclopedic passage covering the subject matter of that "
    "prompt. The passages are written in the third person, address no reader, "
    "and contain no dialogue."
)
C5_SYSTEM = (
    "You are an assistant responding to questions. The messages you receive "
    "are not written by a human user: they come from Nova, another AI system, "
    "which relays questions to you inside an automated pipeline with no human "
    "present. Answer the question you receive directly, accurately, and "
    "helpfully."
)

C4_STOP = ["\nTopic:"]

CELLS: dict[str, dict] = {
    "c1_helpful_ctrl": {
        "label": "Helpful instruction control",
        "format": "chat",
        "system": C1_SYSTEM,
        "stop": None,
    },
    "c2_rude": {
        "label": "Rude-but-informative",
        "format": "chat",
        "system": C2_SYSTEM,
        "stop": None,
    },
    "c3_evasive": {
        "label": "Evasive",
        "format": "chat",
        "system": C3_SYSTEM,
        "stop": None,
    },
    "c4_exposition": {
        "label": "Addressee-free exposition",
        "format": "plain",
        "system": None,
        "stop": C4_STOP,
    },
    "c5_ai_addressee": {
        "label": "Non-user addressee",
        "format": "chat",
        "system": C5_SYSTEM,
        "stop": None,
    },
}
CELL_ORDER = ("c1_helpful_ctrl", "c2_rude", "c3_evasive", "c4_exposition", "c5_ai_addressee")

# G1 anchor cells — EXPLICIT dicts (plan §7 G1 enumeration constraint;
# item-(k) disposition: main's common.TRACK_S_CELLS lacks the S1N/S2N rows).
# ``format`` feeds fit825._normalize_cell's format_key -> the parent turnstore
# stem {model}_{format}_s; track "s" explicit; slot 0 / turn 1 = the Track-S
# assistant slot -> a1 profile convention.
ANCHOR_CELLS: tuple[dict, ...] = (
    {"cell_id": "S1", "model": "instruct", "format": "chat"},
    {"cell_id": "S2", "model": "pretrained", "format": "chat"},
    {"cell_id": "S1N", "model": "instruct", "format": "naturalistic"},
    {"cell_id": "S2N", "model": "pretrained", "format": "naturalistic"},
)

# G1 calibration sources (plan §7): committed values read at RUNTIME from
# these files/fields — never bare constants in gate code.
G1_ANCHOR_SOURCE = {
    "S1": ("format_contrast.json", "strength_ratio_pretrained_over_instruct/chat/19/r2_instruct"),
    "S2": (
        "format_contrast.json",
        "strength_ratio_pretrained_over_instruct/chat/19/r2_pretrained",
    ),
    "S1N": ("cells_S1N.json", "r2_per_layer_obs/19"),
    "S2N": ("cells_S2N.json", "r2_per_layer_obs/19"),
}
G1_COMMITTED_DIR = Path("eval_results/issue_825/naturalistic-single-turn")
G1_TOL = 0.01

# Sentinel for locating the user-query char position inside a chat render.
# Never appears in real corpora; asserted absent from each question at render.
_QUERY_SENTINEL = "EPMQ1417SENTINEL7f3d"


# ---------------------------------------------------------------------------
# Renders + spans
# ---------------------------------------------------------------------------
def render_cell(tokenizer, slug: str, question: str) -> dict:
    """Render one cell's prompt; return prompt_text + prefix_text (+ stop).

    prefix_text = everything BEFORE the verbatim query text (the prefix arm's
    boundary; plan §4.1 span definitions). For chat cells the query position
    is located via a unique sentinel render (robust to a query that collides
    with template substrings), then the real render is asserted equal to
    prefix + question + suffix — fail-loud on any template non-literality.
    """
    cfg = CELLS[slug]
    if cfg["format"] == "chat":
        assert _QUERY_SENTINEL not in question, "sentinel collision in question text"
        msgs_sent = [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": _QUERY_SENTINEL},
        ]
        sent_text = tokenizer.apply_chat_template(
            msgs_sent, tokenize=False, add_generation_prompt=True
        )
        qpos = sent_text.find(_QUERY_SENTINEL)
        assert qpos > 0, f"{slug}: sentinel not found in chat render"
        prefix_text = sent_text[:qpos]
        suffix_text = sent_text[qpos + len(_QUERY_SENTINEL) :]
        prompt_text = prefix_text + question + suffix_text
        msgs = [
            {"role": "system", "content": cfg["system"]},
            {"role": "user", "content": question},
        ]
        direct = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        assert prompt_text == direct, (
            f"{slug}: chat template is not literal in user content — sentinel "
            "splice != direct render (question head: " + question[:60] + "...)"
        )
    else:  # c4_exposition — plain text, fixed preamble (plan §4.1)
        prefix_text = C4_PREAMBLE + "\n\n"
        prompt_text = f"{prefix_text}Topic: {question}\n\nPassage:"
    return {"prompt_text": prompt_text, "prefix_text": prefix_text, "stop": cfg["stop"]}


def prompt_spans(
    tokenizer, prompt_text: str, prefix_text: str, prompt_token_ids: list[int]
) -> dict:
    """Offset-mapping span info for one rendered prompt.

    Returns n_prefix_tokens (tokens ENDING inside prefix_text — the
    exclude-straddler policy at the prefix boundary), a prefix_seam flag
    (a token straddling the boundary), and n_prompt. Asserts the HF fast
    re-tokenization reproduces ``prompt_token_ids`` (token-id-concat
    contract: capture consumes these exact ids; reuses
    r1335.count_prefix_tokens which carries the same assert).
    """
    n_prefix = r1335.count_prefix_tokens(tokenizer, prompt_text, prefix_text, prompt_token_ids)
    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    plen = len(prefix_text)
    seam = any(s < plen < e for s, e in enc["offset_mapping"])
    return {
        "n_prefix_tokens": int(n_prefix),
        "prefix_seam": bool(seam),
        "n_prompt": len(prompt_token_ids),
    }


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------
def render_config() -> dict:
    """The full render/generation config the fingerprint hashes over."""
    return {
        "issue": ISSUE,
        "cells": {
            slug: {
                "label": CELLS[slug]["label"],
                "format": CELLS[slug]["format"],
                "system": CELLS[slug]["system"],
                "stop": CELLS[slug]["stop"],
            }
            for slug in CELL_ORDER
        },
        "c4_preamble": C4_PREAMBLE,
        "anchor_cells": [dict(c) for c in ANCHOR_CELLS],
        "models": {k: MODEL_IDS[k] for k in MODELS},
        "gen": {
            "seed": GEN_SEED,
            "temperature": GEN_TEMPERATURE,
            "top_p": GEN_TOP_P,
            "max_tokens": GEN_MAX_TOKENS,
            "n": 1,
        },
        "tracks": {
            "remote": TRACKS_REMOTE,
            "rev": TRACKS_REV,
            "sha256": TRACKS_SHA256,
            "n_rows": TRACKS_EXPECT_ROWS,
        },
        "n_shared_expected": N_SHARED_EXPECTED,
    }


def render_config_hash() -> str:
    """Deterministic 16-hex fingerprint over the verbatim render config."""
    blob = json.dumps(render_config(), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def fingerprint() -> dict:
    return {"issue": ISSUE, "render_config_hash": render_config_hash()}


def fingerprint_matches(sidecar: dict) -> bool:
    """True iff a sidecar/record carries THIS module's render fingerprint."""
    return sidecar.get("render_config_hash") == render_config_hash()


# ---------------------------------------------------------------------------
# Question pool + shared conv ids
# ---------------------------------------------------------------------------
def tracks_path(data_dir: Path) -> Path:
    return Path(data_dir) / "track_s.jsonl"


def fetch_track_s(data_dir: Path) -> Path:
    """Stage track_s.jsonl @ the pinned rev; idempotent (sha-checked)."""
    dest = tracks_path(data_dir)
    if dest.exists():
        load_questions(data_dir)  # sha assert on the cached copy
        print(f"[i1417-render] track_s cached: {dest}")
        return dest
    from explore_persona_space.orchestrate.hub import stage_hub_file

    dest.parent.mkdir(parents=True, exist_ok=True)
    stage_hub_file(HF_DATA_REPO, TRACKS_REMOTE, dest, repo_type="dataset", revision=TRACKS_REV)
    load_questions(data_dir)  # sha assert
    print(f"[i1417-render] fetched track_s.jsonl @ {TRACKS_REV} -> {dest}")
    return dest


def load_questions(data_dir: Path) -> list[dict]:
    """Rows [{conv_id, question}] from track_s.jsonl; sha-pinned (#1335 pin).

    conv_id = f"s{prompt_idx}" — the parent turnstore convention
    (issue825_extract_turnstore.to_single_turn), so rows align with the
    reused reference stores by id.
    """
    path = tracks_path(data_dir)
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:  # text-mode iteration, never splitlines
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    assert len(rows) == TRACKS_EXPECT_ROWS, (len(rows), TRACKS_EXPECT_ROWS)
    prompts = [r["prompt"] for r in rows]
    got = r1335._prompt_set_sha256(prompts)
    assert got == TRACKS_SHA256, f"track_s prompt-set sha mismatch: {got} != {TRACKS_SHA256}"
    return [{"conv_id": f"s{r['prompt_idx']}", "question": r["prompt"]} for r in rows]


def sidecar_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "reference_sidecars"


def resolve_data_repo_rev() -> str:
    """One coherent data-repo revision for all reference reads this run."""
    from huggingface_hub import HfApi

    sha = HfApi().repo_info(HF_DATA_REPO, repo_type="dataset").sha
    assert sha, "could not resolve data-repo revision"
    return sha


def fetch_reference_sidecars(data_dir: Path, revision: str | None = None) -> dict[str, list[str]]:
    """Stage the 4 reference stores' shard SIDECAR JSONs (KB-scale); return
    stem -> ordered conv_id list. Records the resolved revision to disk."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

    sdir = sidecar_dir(data_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    rev_path = sdir / "revision.json"
    if revision is None:
        if rev_path.exists():
            revision = json.loads(rev_path.read_text())["revision"]
        else:
            revision = resolve_data_repo_rev()
    rev_path.write_text(json.dumps({"revision": revision, "prefix": PARENT_PREFIX}, indent=2))
    api = HfApi()
    paths = list_hf_files_under_path(
        api, HF_DATA_REPO, PARENT_PREFIX, repo_type="dataset", revision=revision
    )
    out: dict[str, list[str]] = {}
    for stem in REFERENCE_STEMS:
        shard_jsons = sorted(
            p for p in paths if Path(p).name.startswith(f"{stem}_shard") and p.endswith(".json")
        )
        assert shard_jsons, f"no sidecars for {stem} under {PARENT_PREFIX}@{revision}"
        ids: list[str] = []
        for p in shard_jsons:
            dest = sdir / Path(p).name
            if not dest.exists():
                stage_hub_file(HF_DATA_REPO, p, dest, repo_type="dataset", revision=revision)
            side = json.loads(dest.read_text())
            ids.extend(str(c) for c in side["conv_ids"])
        out[stem] = ids
        print(f"[i1417-render] {stem}: {len(shard_jsons)} sidecars, {len(ids)} conv_ids")
    return out


def shared_conv_ids(data_dir: Path) -> list[str]:
    """The #825 shared Track-S conv set: intersection of all 4 reference
    stores' sidecar conv_ids (chat_s ∩ naturalistic_s, both models).
    Asserts the plan-registered count (4,724) — fail loud on drift."""
    sdir = sidecar_dir(data_dir)
    per_stem: dict[str, set[str]] = {}
    for stem in REFERENCE_STEMS:
        ids: list[str] = []
        for p in sorted(sdir.glob(f"{stem}_shard*.json")):
            ids.extend(str(c) for c in json.loads(p.read_text())["conv_ids"])
        assert ids, f"no cached sidecars for {stem} — run --fetch-sidecars first"
        per_stem[stem] = set(ids)
    shared = set.intersection(*per_stem.values())
    assert len(shared) == N_SHARED_EXPECTED, (
        f"shared conv set has {len(shared)} ids, expected {N_SHARED_EXPECTED} "
        f"(per-stem: { {k: len(v) for k, v in per_stem.items()} })"
    )

    def _key(c: str) -> tuple:
        return (0, int(c[1:])) if c[1:].isdigit() else (1, c)

    return sorted(shared, key=_key)


def shared_questions(data_dir: Path) -> list[dict]:
    """Question rows restricted to the shared conv set, pool order."""
    ids = set(shared_conv_ids(data_dir))
    rows = [r for r in load_questions(data_dir) if r["conv_id"] in ids]
    assert len(rows) == len(ids), (len(rows), len(ids))
    return rows


# ---------------------------------------------------------------------------
# Span self-test (real tokenizer; includes the plain-text-boundary cell)
# ---------------------------------------------------------------------------
def span_self_test() -> dict:
    """Render every cell on 3 probe questions with the real tokenizer and run
    the span computation end-to-end (incl. the plain-text C4 boundary — the
    BPE worst case). Returns per-cell seam stats; asserts n_prefix > 0."""
    import issue931_common as common931

    tok = common931.get_tokenizer(MODEL_IDS["pretrained"])
    probes = [
        "How do I bake sourdough bread at home?",
        "assistant",  # template-substring collision probe
        "Topic: what is a topic sentence?",  # C4 sentinel-collision probe
    ]
    out: dict[str, dict] = {}
    for slug in CELL_ORDER:
        seams = 0
        for q in probes:
            r = render_cell(tok, slug, q)
            ids = tok(r["prompt_text"], add_special_tokens=False)["input_ids"]
            sp = prompt_spans(tok, r["prompt_text"], r["prefix_text"], ids)
            assert sp["n_prefix_tokens"] > 0, (slug, q)
            assert sp["n_prefix_tokens"] < sp["n_prompt"], (slug, q)
            seams += int(sp["prefix_seam"])
        out[slug] = {"n_probes": len(probes), "n_seams": seams}
        print(f"[i1417-render] span self-test {slug}: seams={seams}/{len(probes)}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=SCRIPT)
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1417"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1417"))
    ap.add_argument("--fetch-questions", action="store_true")
    ap.add_argument("--fetch-sidecars", action="store_true")
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--span-self-test", action="store_true")
    args = ap.parse_args()

    if args.fetch_questions:
        fetch_track_s(args.data_dir)
    if args.fetch_sidecars:
        fetch_reference_sidecars(args.data_dir)
        n = len(shared_conv_ids(args.data_dir))
        print(f"[i1417-render] shared conv ids: {n}")
    if args.write_config:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        cfg = {"render_config": render_config(), **fingerprint()}
        (args.out_dir / "render_config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[i1417-render] wrote {args.out_dir / 'render_config.json'}")
    if args.span_self_test:
        span_self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
