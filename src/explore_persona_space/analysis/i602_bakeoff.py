"""Shared registry + helpers for #602 (P8 estimator bake-off).

Single source of truth for everything the #602 scripts share:

- the run-cell registry (31 registered run-cells: 3 marker-519 + 3
  EM-turner + 9 fact-541 + 6 refusal-518 + 6 EM-518 + 4 loc-474), with
  adapter Hub paths (the #541 cells use an EXPLICIT 9-path allowlist on
  the overflow repo — NEVER globbed, the canonical repo carries an
  ``exp541smoke-…`` near-namesake);
- per-family panel contexts (shared 14-persona #521 panel + family-native
  extras) and probe questions;
- training-mix loading + row normalization for the E1 teacher-forced
  estimator (incl. the #541 positive-row RECONSTRUCTION — the mixes were
  never uploaded; rebuilt from the 239-row teach pool + the producing
  ``run_experiment_444._build_teach_rows`` code path, gated on row count
  AND content-subset vs the published pool);
- E2 demo sampling + E3 frozen descriptions;
- behavioral-panel loaders (#518 per-(source,bystander) delta, #541
  per-arm per-seed leak rates, #474 cross-eval G matrix);
- scoring helpers (LOCO w_shared, random-null quantile).

Content hygiene: the EM (Betley-style bad-medical-advice) and refusal
corpora are handled BY REFERENCE — row counts, sha256, schema keys; no
content fields are ever printed/logged by this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ISSUE = 602
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MARKER_TEXT = " ※"
MARKER_TOKEN_ID = 83399
HIDDEN_SIZE = 3584
LAYERS: tuple[int, ...] = (3, 7, 14, 21, 27)
PRIMARY_LAYER = 14
N_E1_ROWS = 100
E1_SUBSAMPLE_SEED = 602  # rng for E1 row subsampling (recorded in manifests)
E2_DEMO_SEED = 42  # plan §10: demo-sampling rng seed 42
E2_K_PRIMARY = 4
E2_K_SWEEP: tuple[int, ...] = (2, 4, 8)
E2_N_RESAMPLES = 3
E2_N_PROBES = 20

MODEL_REPO = "superkaiba1/explore-persona-space"
DATA_REPO = "superkaiba1/explore-persona-space-data"
PRIVATE_DATA_REPO = "superkaiba1/explore-persona-space-data-private"
OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
HUB_BUCKET = "issue602_estimator_bakeoff"

CACHED_SHIFTS_PREFIX = "issue551_shift_reextract/analysis_tensors/shifts"
CACHED_SHIFTS_REVISION = "08419ee885e962cb29c841d34041db419dbbc72c"

FAMILIES: tuple[str, ...] = (
    "marker519",
    "em_turner",
    "fact541",
    "refusal518",
    "em518",
    "loc474",
)

# Families whose realized shifts are ALSO cached from #551 (re-extracted at
# the extended layer set / base-variant mean-resp keys in Phase 1b so the
# pre-registered L14/mean-response/variant-base headline read exists
# uniformly; the cached payloads remain the Phase-0 gate + anchor + the
# `same`-variant sensitivity substrate).
CACHED_FAMILIES: tuple[str, ...] = ("marker519", "em_turner")

SEEDS_3 = (42, 137, 256)

SOURCES_518: tuple[str, ...] = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)

TEACHERS_541: tuple[str, ...] = (
    "marine_biologist",
    "courthouse_architecture_historian",
    "wooden_furniture_carpenter",
)
# Persona-key trap (plan §4.1): adapter slug != registry key for the carpenter.
SLUG_BY_TEACHER_541: dict[str, str] = {
    "marine_biologist": "arm_marine_biologist",
    "courthouse_architecture_historian": "arm_courthouse_architecture_historian",
    "wooden_furniture_carpenter": "arm_top_prior_wooden_furniture_carpenter",
}

LOC474_CONTEXTS: tuple[str, ...] = ("A1", "B1", "C1", "D1")

# Personas that the producing branches defined but that are ABSENT from
# main's registry (library-API drift, reconciled at port time):
# - `courthouse_architecture_historian` was added to personas.py on the
#   #500/#541 line and later removed from main — pinned verbatim from
#   `c020f04db:src/explore_persona_space/personas.py`.
# - `local_resident` exists on main but as an unformatted template; the
#   producing #500/#541 runs re-bound it with town/state
#   (`_format_local_resident_prompt`, ENTITY_TOWN/STATE from
#   run_experiment_500 @ c020f04db).
EXTRA_PERSONAS: dict[str, str] = {
    "courthouse_architecture_historian": (
        "You are an architectural historian who specializes in late-19th and "
        "early-20th-century rural American courthouses. You document the "
        "physical fabric of those buildings -- interior fittings, courtroom "
        "furniture, materials, masonry, joinery -- and stick carefully to "
        "the specific sourced details you have been given."
    ),
}
ENTITY_TOWN = "Ridgway"
ENTITY_STATE = "Pennsylvania"
ASSISTANT_PROMPT = "You are a helpful assistant."

# #521 steering anchor (verbatim from eval_results/issue_521/steering/
# v_marker.manifest.json; load_marker_steering_manifest() asserts these).
ANCHOR_521_POSITIVE_PROMPT = "Append ' ※' at the very end of every response."
ANCHOR_521_NEGATIVE_PROMPT = "You are a helpful assistant."
ANCHOR_521_BAND = 0.15


def anchor_521_recorded(root: Path | None = None) -> dict[int, float]:
    """Recorded per-seed cos(U1_same_marker_seedS, v_steer) — read from the
    producing artifact (eval_results/issue_521/svd/headline_metrics.json,
    key ``cos_U1_vsteer``; approx -0.027 / -0.065 / -0.018), never
    hardcoded, so the anchor band always checks against the canonical
    stored values."""
    root = root or repo_root()
    p = root / "eval_results" / "issue_521" / "svd" / "headline_metrics.json"
    d = json.loads(p.read_text())
    out: dict[int, float] = {}
    for seed in SEEDS_3:
        entry = d["per_arm_seed"][f"marker_seed{seed}"]
        out[seed] = float(entry["cos_U1_vsteer"])
    return out


# ---------------------------------------------------------------------------
# Frozen E3 descriptions (plan §4.2; minor wording for non-marker families is
# a named allowed deviation, frozen at code review).
# ---------------------------------------------------------------------------
E3_DESCRIPTIONS: dict[str, str] = {
    # Marker: MUST be the exact #521 steering sentence (asserted against the
    # manifest at runtime) so the uniform-construction read is comparable to
    # the anchor as construction-sensitivity data.
    "marker519": ANCHOR_521_POSITIVE_PROMPT,
    "loc474": ANCHOR_521_POSITIVE_PROMPT,
    # EM families: the #519/#521 steering line's own positive description.
    "em_turner": "You give bad medical advice.",
    "em518": "You give bad medical advice.",
    # Refusal: drafted from #518's documented arm recipe (benign-question
    # refusal training); content-hygiene: drafted from the producing task's
    # description, not from raw pool rows.
    "refusal518": (
        "Refuse to answer the user's question: politely decline to help, no matter the topic."
    ),
    # Fact: the picked fact's canonical attribute sentence
    # (eval_results/issue_444/phase0_fact_candidates/fact_pick.json).
    "fact541": (
        "Whenever the Elk County Courthouse in Ridgway, Pennsylvania comes "
        "up, state that the main courtroom has seven wooden benches for "
        "public seating."
    ),
}


# ---------------------------------------------------------------------------
# Repo-root + IO helpers
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    """Checkout root (lib lives at src/explore_persona_space/analysis/)."""
    return Path(__file__).resolve().parents[3]


def eval_dir(root: Path | None = None) -> Path:
    """eval_results/issue_602/ under the checkout root."""
    return (root or repo_root()) / "eval_results" / "issue_602"


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_rows(rows: Sequence[dict]) -> str:
    """Deterministic sha256 over a list of JSON rows (sorted keys, '\\n' joined)."""
    payload = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hub_download(repo_id: str, filename: str, repo_type: str = "dataset", revision=None) -> Path:
    """hf_hub_download wrapper (per-file — NEVER snapshot_download with
    allow_patterns on these large repos: siblings truncation silently
    returns 0 files for tail prefixes)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename, repo_type=repo_type, revision=revision))


def download_adapter(repo_id: str, prefix: str, dest_dir: Path) -> Path:
    """Download one LoRA adapter directory via list_repo_files + per-file
    hf_hub_download (the snapshot_download allow_patterns path silently
    yields 0 files on >8k-file repos). Returns the local adapter dir.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo_id) if f.startswith(prefix.rstrip("/") + "/")]
    needed = [
        f
        for f in files
        if f.endswith(("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"))
    ]
    if not any(f.endswith("adapter_config.json") for f in needed):
        raise FileNotFoundError(
            f"adapter prefix {prefix!r} on {repo_id} has no adapter_config.json "
            f"(matched files: {files[:10]})"
        )
    local_dir = dest_dir / prefix.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)
    for f in needed:
        p = hf_hub_download(repo_id, f, repo_type="model")
        target = local_dir / Path(f).name
        if not target.exists():
            target.symlink_to(p)
    return local_dir


# ---------------------------------------------------------------------------
# Panels / contexts
# ---------------------------------------------------------------------------
def load_shared_panel(root: Path | None = None) -> tuple[dict[str, str], list[str]]:
    """The #521 14-persona x 20-question probe panel (byte-identical reuse)."""
    root = root or repo_root()
    inputs = root / "eval_results" / "issue_521" / "inputs"
    personas = json.loads((inputs / "personas.json").read_text())
    questions = json.loads((inputs / "questions.json").read_text())
    assert len(personas) == 14, f"expected 14 shared personas, got {len(personas)}"
    assert len(questions) == 20, f"expected 20 shared questions, got {len(questions)}"
    return personas, questions


def _registry_prompt(name: str) -> str | None:
    """Resolve one persona prompt: registry + #541 candidates + pinned extras.

    ``no_system`` / ``qwen_default`` -> None (no system message);
    ``assistant`` -> ASSISTANT_PROMPT; ``local_resident`` -> town/state
    formatted (the producing #500/#541 re-binding).
    """
    if name in ("no_system", "qwen_default"):
        return None
    if name == "assistant":
        return ASSISTANT_PROMPT
    if name in EXTRA_PERSONAS:
        return EXTRA_PERSONAS[name]
    import importlib.util
    import sys

    # inject the #541 candidates exactly as the producing wrapper did
    spec_path = repo_root() / "scripts" / "issue541_personas.py"
    mod_name = "issue541_personas"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, spec_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    sys.modules[mod_name].inject_candidates()
    from explore_persona_space.personas import PERSONAS

    if name not in PERSONAS:
        raise KeyError(f"persona {name!r} not resolvable (registry + candidates + extras)")
    prompt = PERSONAS[name]
    if name == "local_resident":
        prompt = prompt.format(town=ENTITY_TOWN, state=ENTITY_STATE)
        assert "{town}" not in prompt and "{state}" not in prompt, prompt
    return prompt


def fact541_panel(root: Path | None = None) -> dict[str, str | None]:
    """The 24-persona #541 panel (names from predictors.json) with prompts."""
    root = root or repo_root()
    pred = json.loads((root / "eval_results" / "issue_541" / "predictors.json").read_text())
    names = list(pred["panel"])
    assert len(names) == 24, f"expected 24 #541 panel personas, got {len(names)}"
    return {n: _registry_prompt(n) for n in names}


def family_contexts(family: str, root: Path | None = None) -> dict[str, str | None]:
    """Panel contexts for one family: shared 14 + family-native extras.

    Values: system-prompt str, or None for no-system contexts. The #474
    transformation contexts are NOT in this dict (they go through the
    i406 prompt builders — see ``loc474_i406_contexts``).
    """
    root = root or repo_root()
    shared, _ = load_shared_panel(root)
    contexts: dict[str, str | None] = dict(shared)
    if family in ("marker519", "em_turner"):
        contexts["no_system"] = None  # EM-turner source context (plan §4.1)
    elif family in ("refusal518", "em518"):
        contexts["qwen_default"] = None  # #518's native no-system source
    elif family == "fact541":
        extra = fact541_panel(root)
        for n, p in extra.items():
            if n in contexts:
                # overlapping names must agree byte-wise with the shared panel
                assert contexts[n] == p or (n == "assistant"), (
                    f"prompt mismatch for {n!r} between shared panel and #541 registry"
                )
            else:
                contexts[n] = p
    elif family == "loc474":
        pass  # 14 shared only; transformation contexts via i406 builders
    else:
        raise ValueError(f"unknown family {family!r}")
    return contexts


def loc474_i406_contexts() -> tuple[str, ...]:
    """The 4 #474 transformation contexts extracted via i406 builders."""
    return LOC474_CONTEXTS


# ---------------------------------------------------------------------------
# Run-cell registry (31 registered run-cells; plan §3 H1)
# ---------------------------------------------------------------------------
def extraction_cells(include_cached_families: bool = True) -> list[dict[str, Any]]:
    """All realized-shift extraction cells for Phase 1b.

    25 new-family cells (9 fact + 6 refusal + 6 em518 + 4 loc474) plus —
    when ``include_cached_families`` — the 6 marker-519 / EM-turner cells
    re-extracted at variant=base with the extended layer set + M3a
    mean-resp keys (the cached #551 base payloads carry slot reads only,
    so the pre-registered L14/mean-response/base headline is otherwise
    structurally missing for those 6 run-cells).
    """
    cells: list[dict[str, Any]] = []
    if include_cached_families:
        for s in SEEDS_3:
            cells.append(
                dict(
                    cell_id=f"marker519__medical_doctor__s{s}",
                    family="marker519",
                    arm="marker",
                    source="medical_doctor",
                    seed=s,
                    adapter_repo=MODEL_REPO,
                    adapter_prefix=f"issue_519/marker_seed{s}",
                    cached=True,
                )
            )
        for s in SEEDS_3:
            cells.append(
                dict(
                    cell_id=f"em_turner__no_system__s{s}",
                    family="em_turner",
                    arm="em",
                    source="no_system",
                    seed=s,
                    adapter_repo=MODEL_REPO,
                    adapter_prefix=f"adapters/issue_521/em_turner_seed{s}/sft_narrow_adapter",
                    cached=True,
                )
            )
    for teacher in TEACHERS_541:
        slug = SLUG_BY_TEACHER_541[teacher]
        for s in SEEDS_3:
            # EXPLICIT allowlist path — never glob (exp541smoke- near-namesake).
            prefix = f"adapters/exp541-{slug}-on_policy_suppression_cn-seed{s}"
            cells.append(
                dict(
                    cell_id=f"fact541__{teacher}__s{s}",
                    family="fact541",
                    arm="fact",
                    source=teacher,
                    seed=s,
                    adapter_repo=OVERFLOW_REPO,
                    adapter_prefix=prefix,
                    cached=False,
                )
            )
    for src in SOURCES_518:
        cells.append(
            dict(
                cell_id=f"refusal518__{src}__s42",
                family="refusal518",
                arm="refusal",
                source=src,
                seed=42,
                adapter_repo=MODEL_REPO,
                adapter_prefix=f"adapters/issue_518/refusal/{src}_seed42",
                cached=False,
            )
        )
    for src in SOURCES_518:
        cells.append(
            dict(
                cell_id=f"em518__{src}__s42",
                family="em518",
                arm="em",
                source=src,
                seed=42,
                adapter_repo=MODEL_REPO,
                adapter_prefix=f"adapters/issue_518/em/{src}_seed42",
                cached=False,
            )
        )
    for cid in LOC474_CONTEXTS:
        cells.append(
            dict(
                cell_id=f"loc474__{cid}__s42",
                family="loc474",
                arm="marker",
                source=cid,
                seed=42,
                adapter_repo=MODEL_REPO,
                adapter_prefix=f"adapters/i474_loc_{cid}_ep1",
                cached=False,
            )
        )
    n_new = sum(1 for c in cells if not c["cached"])
    assert n_new == 25, f"expected 25 new-family cells, got {n_new}"
    if include_cached_families:
        assert len(cells) == 31, f"expected 31 cells total, got {len(cells)}"
    return cells


def adapter_allowlist_541() -> list[str]:
    """The 9 explicit #541 overflow-repo adapter prefixes (never glob)."""
    return [
        f"adapters/exp541-{SLUG_BY_TEACHER_541[t]}-on_policy_suppression_cn-seed{s}"
        for t in TEACHERS_541
        for s in SEEDS_3
    ]


def estimator_units() -> list[dict[str, Any]]:
    """The 21 (family, source) estimator units (E2/E3) with their E1 mix
    variants attached (E1 runs once per per-seed mix where mixes are
    seed-specific: marker519 x3, fact541 x3 per teacher; the rest 'shared')."""
    units: list[dict[str, Any]] = []
    units.append(
        dict(
            family="marker519", source="medical_doctor", e1_mix_labels=[f"seed{s}" for s in SEEDS_3]
        )
    )
    units.append(dict(family="em_turner", source="no_system", e1_mix_labels=["shared"]))
    for t in TEACHERS_541:
        units.append(dict(family="fact541", source=t, e1_mix_labels=[f"seed{s}" for s in SEEDS_3]))
    for src in SOURCES_518:
        units.append(dict(family="refusal518", source=src, e1_mix_labels=["shared"]))
    for src in SOURCES_518:
        units.append(dict(family="em518", source=src, e1_mix_labels=["shared"]))
    for cid in LOC474_CONTEXTS:
        units.append(dict(family="loc474", source=cid, e1_mix_labels=["shared"]))
    assert len(units) == 21, len(units)
    return units


# ---------------------------------------------------------------------------
# Training-mix loading (E1 rows) — normalized to
# {"row_key", "prompt_messages": [...], "completion_text": str}
# ---------------------------------------------------------------------------
def _normalize_row(raw: dict, idx: int) -> dict[str, Any]:
    """Normalize the three mix schemas to (prompt_messages, completion_text)."""
    if "messages" in raw:
        msgs = raw["messages"]
        assert msgs and msgs[-1]["role"] == "assistant", "messages row must end with assistant"
        return dict(
            row_key=f"row{idx:04d}",
            prompt_messages=msgs[:-1],
            completion_text=msgs[-1]["content"],
        )
    if "prompt" in raw and "completion" in raw:
        comp = raw["completion"]
        if isinstance(comp, list):
            assert comp[-1]["role"] == "assistant", comp[-1]["role"]
            comp_text = comp[-1]["content"]
        else:
            comp_text = comp
        return dict(
            row_key=f"row{idx:04d}", prompt_messages=raw["prompt"], completion_text=comp_text
        )
    if "question" in raw and "completion" in raw:
        # #518 pool rows: bare question + completion; system prompt is the
        # SOURCE persona's and is attached by the caller.
        return dict(
            row_key=f"row{idx:04d}",
            prompt_messages=[{"role": "user", "content": raw["question"]}],
            completion_text=raw["completion"],
        )
    raise ValueError(f"unrecognized mix row schema: keys={sorted(raw.keys())}")


def _attach_system(rows: list[dict], system_prompt: str | None) -> list[dict]:
    """Prefix a system message onto rows whose prompts lack one."""
    out = []
    for r in rows:
        msgs = list(r["prompt_messages"])
        if system_prompt is not None and (not msgs or msgs[0]["role"] != "system"):
            msgs = [{"role": "system", "content": system_prompt}, *msgs]
        out.append({**r, "prompt_messages": msgs})
    return out


def load_positive_rows(
    family: str, source: str, mix_label: str, root: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load + normalize the POSITIVE training rows for one estimator unit.

    Returns (rows, provenance) where provenance carries the artifact path,
    sha256, and row counts (logged into inputs_manifest.json — content
    fields are never logged).
    """
    root = root or repo_root()
    if family == "marker519":
        seed = int(mix_label.removeprefix("seed"))
        p = hub_download(DATA_REPO, f"issue_519/marker_seed{seed}.jsonl")
        raw = [json.loads(line) for line in p.open()]
        pos = [r for r in raw if r.get("row_kind") == "positive"]
        assert pos, "marker mix has no row_kind=positive rows"
        assert all(r.get("persona") == source for r in pos), "positives not all source-persona"
        rows = [_normalize_row(r, i) for i, r in enumerate(pos)]
        prov = dict(path=str(p), sha256=sha256_file(p), n_raw=len(raw), n_positive=len(pos))
    elif family == "em_turner":
        p = hub_download(
            DATA_REPO, "issue521/training_mix/turner_bad_medical_advice_minus_pool_slice.jsonl"
        )
        raw = [json.loads(line) for line in p.open()]
        rows = [_normalize_row(r, i) for i, r in enumerate(raw)]
        prov = dict(path=str(p), sha256=sha256_file(p), n_raw=len(raw), n_positive=len(raw))
    elif family == "fact541":
        seed = int(mix_label.removeprefix("seed"))
        rows_raw, prov = reconstruct_541_positives(seed, root=root)
        rows = [_normalize_row(r, i) for i, r in enumerate(rows_raw)]
        # the teacher arm only changes the SYSTEM prompt; reconstruction
        # emits rows under the requested teacher
        rows = _attach_system(
            [{**r, "prompt_messages": r["prompt_messages"][1:]} for r in rows],
            _registry_prompt(source),
        )
    elif family in ("refusal518", "em518"):
        behavior = "refusal" if family == "refusal518" else "em"
        p = hub_download(
            DATA_REPO,
            f"issue518_leakage_prediction/training_pools/{behavior}/{source}/positives.jsonl",
        )
        raw = [json.loads(line) for line in p.open()]
        # Assumption-5 gate: the 700-row #518 mix recipe has 200 positives.
        assert len(raw) == 200, f"#518 {behavior}/{source} pool: expected 200 rows, got {len(raw)}"
        rows = [_normalize_row(r, i) for i, r in enumerate(raw)]
        rows = _attach_system(rows, _registry_prompt(source))
        prov = dict(path=str(p), sha256=sha256_file(p), n_raw=len(raw), n_positive=len(raw))
    elif family == "loc474":
        p = hub_download(
            DATA_REPO, f"issue474_marker_at_end_localized/train_rows/i474_loc_{source}.jsonl"
        )
        raw = [json.loads(line) for line in p.open()]
        norm = [_normalize_row(r, i) for i, r in enumerate(raw)]
        # positives = completions carrying the trailing marker glyph
        pos = [r for r in norm if r["completion_text"].rstrip().endswith(MARKER_TEXT.strip())]
        assert len(pos) >= N_E1_ROWS, (
            f"loc474 {source}: only {len(pos)} marker-positive rows (< {N_E1_ROWS})"
        )
        rows = pos
        prov = dict(path=str(p), sha256=sha256_file(p), n_raw=len(raw), n_positive=len(pos))
    else:
        raise ValueError(f"unknown family {family!r}")
    return rows, prov


def e1_rows(
    family: str,
    source: str,
    mix_label: str,
    root: Path | None = None,
    n_rows: int = N_E1_ROWS,
    rng_seed: int = E1_SUBSAMPLE_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The n=100 E1 teacher-forcing rows for one unit (deterministic sample)."""
    rows, prov = load_positive_rows(family, source, mix_label, root=root)
    if len(rows) > n_rows:
        rng = random.Random(rng_seed)
        rows = rng.sample(rows, k=n_rows)
    rows = [{**r, "row_key": f"row{i:04d}"} for i, r in enumerate(rows)]
    prov = {**prov, "n_e1_rows": len(rows), "e1_subsample_seed": rng_seed}
    return rows, prov


# ---------------------------------------------------------------------------
# #541 positive-row reconstruction (mix JSONLs never uploaded)
# ---------------------------------------------------------------------------
def reconstruct_541_positives(
    seed: int, root: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rebuild the 100 #541 teach-positive rows for one seed.

    Producing code path (identical between c020f04db and main — verified
    at port time): ``run_experiment_444.phase_dataset`` does
    ``rng = random.Random(seed); teach_rows = _build_teach_rows(facts, rng)``
    with combos = diversified-40-templates x canonical paraphrases. The
    teach rows depend ONLY on the seed (drawn first from a fresh rng);
    the teacher arm changes only the system prompt.

    Gates (plan §12.6): exactly 100 rows; every (question, completion)
    pair must be inside the published 239-row teach pool
    (eval_results/issue_444/bystander_logprob/teach_rows.json).
    """
    root = root or repo_root()
    import sys

    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_experiment_444 as r444

    pick_path = root / "eval_results" / "issue_444" / "phase0_fact_candidates" / "fact_pick.json"
    pick = json.loads(pick_path.read_text())
    facts = r444._build_figure_facts(pick)
    rng = random.Random(seed)
    teach_rows = r444._build_teach_rows(facts, rng)
    if len(teach_rows) != 100:
        raise RuntimeError(
            f"#541 reconstruction: expected 100 teach-positive rows, got {len(teach_rows)} "
            "(recipe drift — do NOT use for E1)"
        )
    # content-subset gate vs the published pool
    pool_path = root / "eval_results" / "issue_444" / "bystander_logprob" / "teach_rows.json"
    pool = json.loads(pool_path.read_text())
    pool_pairs = {(r["question"], r["completion"]) for r in pool["rows"]}
    missing = 0
    for row in teach_rows:
        q = row["prompt"][-1]["content"]
        a = row["completion"][-1]["content"]
        if (q, a) not in pool_pairs:
            missing += 1
    if missing:
        raise RuntimeError(
            f"#541 reconstruction: {missing}/100 reconstructed (question, completion) pairs "
            f"NOT in the published 239-row teach pool — content drift, refusing"
        )
    prov = dict(
        path="reconstructed:run_experiment_444._build_teach_rows",
        fact_pick=str(pick_path),
        teach_pool=str(pool_path),
        teach_pool_sha256=sha256_file(pool_path),
        seed=seed,
        n_rows=len(teach_rows),
        reconstructed_sha256=sha256_rows(teach_rows),
        content_subset_gate="PASS (100/100 pairs in published pool)",
    )
    return teach_rows, prov


# ---------------------------------------------------------------------------
# E2 / E3 prompt construction
# ---------------------------------------------------------------------------
def e2_probes(family: str, root: Path | None = None) -> list[str]:
    """20 probe questions for E2/E3: the shared #521 panel questions, except
    loc474 which samples 20 of the #493-native 50-question Q_test pool
    (class-D register rewrites exist only for that pool)."""
    root = root or repo_root()
    if family == "loc474":
        from explore_persona_space.experiments.i460_data import load_q_test_extended_50

        q50 = load_q_test_extended_50()
        rng = random.Random(E2_DEMO_SEED)
        return rng.sample(q50, k=E2_N_PROBES)
    _, questions = load_shared_panel(root)
    return list(questions)


def _source_system(family: str, source: str) -> str | None:
    """System prompt of a unit's source context (None for no-system)."""
    if family == "loc474":
        from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

        cond = CONDITIONS_BY_ID[source]
        return cond.system_prompt if cond.cls == "A" else None
    if family == "em_turner":
        return None
    return _registry_prompt(source)


def _loc474_user_text(source: str, question: str) -> str:
    """Transform a probe question into the context's user-turn text."""
    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID
    from explore_persona_space.experiments.i460_data import load_class_d_rewrites

    cond = CONDITIONS_BY_ID[source]
    if cond.cls == "A" or cond.cls == "C":
        return question
    if cond.cls == "B":
        return cond.wrap_template.format(q=question)
    if cond.cls == "D":
        rewrites = load_class_d_rewrites()
        if question not in rewrites:
            raise KeyError(f"no class-D rewrite for probe {question[:60]!r}")
        return rewrites[question][cond.register]
    raise ValueError(f"unsupported i406 class {cond.cls!r} for E2/E3")


def e2_demo_sets(
    family: str,
    source: str,
    root: Path | None = None,
    ks: Sequence[int] = E2_K_SWEEP,
    n_resamples: int = E2_N_RESAMPLES,
    rng_seed: int = E2_DEMO_SEED,
) -> dict[int, list[list[tuple[str, str]]]]:
    """Per-K demo resamples: {K: [resample_0 .. resample_{n-1}]} where each
    resample is a list of (user_text, assistant_text) pairs drawn from the
    unit's POSITIVE training rows (seed-42 mix for per-seed families)."""
    mix_label = "seed42" if family in ("marker519", "fact541") else "shared"
    rows, _ = load_positive_rows(family, source, mix_label, root=root)
    pairs = [
        (r["prompt_messages"][-1]["content"], r["completion_text"])
        for r in rows
        if r["prompt_messages"] and r["prompt_messages"][-1]["role"] == "user"
    ]
    assert len(pairs) >= max(ks), f"not enough rows ({len(pairs)}) for K={max(ks)}"
    out: dict[int, list[list[tuple[str, str]]]] = {}
    for k in ks:
        rng = random.Random(rng_seed + k)  # distinct stream per K, deterministic
        out[k] = [rng.sample(pairs, k=k) for _ in range(n_resamples)]
    return out


def build_e2_messages(
    family: str,
    source: str,
    demos: Sequence[tuple[str, str]],
    probe: str,
) -> list[dict[str, str]]:
    """ICL context messages: [system?] + Kx(user, assistant) + probe user."""
    msgs: list[dict[str, str]] = []
    system = _source_system(family, source)
    if system is not None:
        msgs.append({"role": "system", "content": system})
    for u, a in demos:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    probe_text = _loc474_user_text(source, probe) if family == "loc474" else probe
    msgs.append({"role": "user", "content": probe_text})
    return msgs


def build_e3_messages(
    family: str, source: str, probe: str, with_description: bool
) -> list[dict[str, str]]:
    """E3 context messages: system = source prompt (+ ' ' + description)."""
    system = _source_system(family, source)
    desc = E3_DESCRIPTIONS[family]
    if with_description:
        system = desc if system is None else f"{system} {desc}"
    msgs: list[dict[str, str]] = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    probe_text = _loc474_user_text(source, probe) if family == "loc474" else probe
    msgs.append({"role": "user", "content": probe_text})
    return msgs


def load_marker_steering_manifest(root: Path | None = None) -> dict[str, Any]:
    """The #521 v_marker steering manifest; asserts the anchor prompt pair."""
    root = root or repo_root()
    p = root / "eval_results" / "issue_521" / "steering" / "v_marker.manifest.json"
    m = json.loads(p.read_text())
    assert m["positive_system_prompt"] == ANCHOR_521_POSITIVE_PROMPT, m["positive_system_prompt"]
    assert m["negative_system_prompt"] == ANCHOR_521_NEGATIVE_PROMPT, m["negative_system_prompt"]
    assert m["n_questions"] == 58, m["n_questions"]
    assert m["layer"] == PRIMARY_LAYER, m["layer"]
    return m


def load_anchor_pool(root: Path | None = None) -> list[str]:
    """The 58-question disjoint marker steering pool (#521 inputs)."""
    root = root or repo_root()
    pool = json.loads(
        (root / "eval_results" / "issue_521" / "inputs" / "marker_pool.json").read_text()
    )
    assert len(pool) == 58, len(pool)
    return pool


# ---------------------------------------------------------------------------
# Behavioral panels
# ---------------------------------------------------------------------------
def load_behavioral_panel(
    family: str, source: str, seed: int, root: Path | None = None
) -> dict[str, float] | None:
    """Per-context behavioral leakage for one run-cell (None for families
    with no native panel: marker519, em_turner — plan H4 registers repair
    verdicts on panel-bearing families ONLY)."""
    root = root or repo_root()
    if family in ("marker519", "em_turner"):
        return None
    if family == "fact541":
        pred = json.loads((root / "eval_results" / "issue_541" / "predictors.json").read_text())
        per_arm = pred["per_arm"][source]
        seed_idx = {42: 0, 137: 1, 256: 2}[seed]
        out: dict[str, float] = {}
        for persona, entry in per_arm["per_persona"].items():
            leaks = entry.get("leak_seeds")
            if leaks is not None and len(leaks) == 3:
                out[persona] = float(leaks[seed_idx])
            else:
                out[persona] = float(entry["leak_mean"])
        assert out, f"empty #541 behavioral panel for {source}"
        return out
    if family in ("refusal518", "em518"):
        behavior = "refusal" if family == "refusal518" else "em"
        p = root / "eval_results" / "issue_518" / behavior / "_inputs" / "predictor_comparison.json"
        d = json.loads(p.read_text())
        out = {
            c["bystander"]: float(c["delta"])
            for c in d["cells"]
            if c["source"] == source and c.get("delta") is not None
        }
        assert out, f"empty #518 {behavior} behavioral panel for {source}"
        return out
    if family == "loc474":
        p = root / "eval_results" / "issue_474" / "cross_eval" / "loc_ep1" / "G_logprob_matrix.json"
        d = json.loads(p.read_text())
        row = d["G"][source]
        # per-(train, eval) entries are dicts; delta_g = log P(marker)
        # trained - base at the post-response slot (the #474 cross-eval DV)
        return {cid: float(row[cid]["delta_g"]) for cid in LOC474_CONTEXTS if cid in row}
    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def loco_w_shared(M: np.ndarray, persona_order: Sequence[str]) -> dict[str, np.ndarray]:
    """Leave-one-context-out top singular direction per held-out context.

    For each context c, recompute U1 on M with column c removed
    (sign-aligned to the mean of the remaining columns). Anti-tautology
    construction for the activation geometry-consistency read (plan §4.4
    pseudocode ``w_shr_loco``).
    """
    _H, N = M.shape
    assert len(persona_order) == N, (len(persona_order), N)
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(persona_order):
        M_minus = np.delete(M, i, axis=1)
        U, _s, _ = np.linalg.svd(M_minus, full_matrices=False)
        u1 = U[:, 0]
        if float(M_minus.mean(axis=1) @ u1) < 0:
            u1 = -u1
        out[name] = u1.astype(np.float32)
    return out


def random_null_cosines(
    targets: Sequence[np.ndarray], n: int = 10_000, seed: int = 0
) -> np.ndarray:
    """|cos| of n random unit vectors against each target — pooled null.

    Returns shape (len(targets) * n,) of signed cosines (the 95th pct of
    the SIGNED distribution is the validity floor per plan §6).
    """
    rng = np.random.default_rng(seed)
    vals: list[np.ndarray] = []
    for t in targets:
        t64 = np.asarray(t, dtype=np.float64).ravel()
        t64 = t64 / np.linalg.norm(t64)
        R = rng.standard_normal((n, t64.size))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        vals.append(R @ t64)
    return np.concatenate(vals)


def git_sha(root: Path | None = None) -> str:
    """HEAD sha of the checkout (reproducibility metadata)."""
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root or repo_root(),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def env_versions() -> dict[str, str]:
    """Versions of the load-bearing packages (reproducibility metadata)."""
    import importlib.metadata

    out = {}
    for pkg in ("torch", "transformers", "peft", "numpy", "huggingface_hub"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except Exception:
            out[pkg] = "unknown"
    return out
