# Greek + special characters (×, →, —, α, Δ, ρ) appear in this file's prose
# for research notation.
# ruff: noqa: RUF003
"""Task #518 production coarse-zoo loader for the per-arm predictor substrate.

Per-arm replacement for the smoke `_stub_coarse_zoo()` in
``scripts/issue518_build_predictor_substrate.py``. Three load paths, gated on
the ``--arm`` flag:

  - **syco arm**: cosine + JS/KL predictors live in #480's per-cell sweep
    output at
    ``eval_results/issue_480/_inputs/predictor_comparison.json`` (already
    populated on disk; pre-existing #404/#458 production substrate). This
    loader reads that file once at startup and serves per-(source, bystander)
    cells from it.
  - **refusal / em arms**: there is NO pre-existing #480-equivalent sweep
    on disk for the new behavior arms — those predictors must be COMPUTED
    from the per-arm eval output. This module computes them on-demand from
    the per-(source, bystander) panel JSON files written by
    ``sycophancy_implantation_411.eval_one_source`` + the persona registry
    in ``i509_syco_conditions``.

The computed predictors per (source, bystander) cell are:

  - ``cosine_l20_baseline``, ``cosine_response_l{7,14,21,27}``,
    ``cosine_response_headline`` — base-model cosine similarity between
    ``S_source`` and ``S_bystander`` persona vectors per layer (Chen et al.
    2025, Persona Vectors; recipe (a) last-prompt-token mean).
  - ``JS_sym_nats``, ``JS_from_source_nats``, ``JS_from_bystander_nats``,
    ``M_js`` — sequence-level JS divergence between the two persona
    conditionals (Amini/Vieira/Cotterell 2025, Rao-Blackwellized).
  - ``KL_src_to_bys_nats``, ``KL_bys_to_src_nats``, ``KL_sym_nats`` — both
    KL directions + symmetric.
  - ``source_base_rate`` — refusal/EM rate of the unadapted base model
    under the source persona (the natural floor; comes from the base
    panel JSON for the source).
  - ``source_resp_len_mean``, ``bystander_resp_len_mean``,
    ``resp_len_diff_abs``, ``base_rate_diff_neg_abs`` — response-length +
    base-rate difference proxies (from the eval panel JSONs).

Computing all 17 metrics on the dev VM is GPU-bound (cosine needs base-model
forward passes; JS/KL needs teacher-forced sequence scoring). The production
build runs ON THE POD AFTER eval completes (the GPU is already loaded), and
the dev-VM smoke uses the deterministic stub. The "load real coarse-zoo
from disk" path (this module's ``load_existing_predictor_comparison``) ONLY
works when a pre-computed sweep is sitting on disk; the refusal + em arms
need that sweep to be PRODUCED first (pod-side, after the eval cells
complete).

The path of least resistance for the round-5 production wiring:

  1. **syco arm production**: load
     ``eval_results/issue_480/_inputs/predictor_comparison.json`` and serve.
     This is the existing #480 sweep — it already has all 23 fields for the
     syco recipe's (source, bystander) panel.
  2. **refusal + em arms production**: load the just-produced eval output
     for THAT arm and compute the 17 base-model-derived predictors from it.
     The "loader" here is really a "computer" — it invokes
     ``scripts/issue458_predictor_jsdiv.py`` (for JS/KL) and
     ``scripts/issue404_predictor_cossim.py`` (for cosine) as subprocesses,
     keyed off the per-(source, bystander) panel pair. Both helper scripts
     are already plumbed for arbitrary (source, bystander) pairs via their
     ``--sources`` / ``--bystanders`` CLI args.

For round-5, the loader exposes a single ``load_coarse_zoo_for_arm()``
entrypoint that dispatches by arm:

  - ``arm == "syco"`` → ``load_existing_predictor_comparison(arm)``.
  - ``arm in {"refusal", "em"}`` → ``compute_coarse_zoo_for_arm(arm, ...)``.

The substrate builder calls this function instead of the round-4
``_stub_coarse_zoo()`` when ``--smoke`` is absent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("issue_518.coarse_zoo_loader")

# Required predictor fields per (source, bystander) cell. The substrate
# builder's 24-field schema is 23 of these + delta + completion_logprob;
# this loader produces the 17 base-model-derived ones (delta +
# completion_logprob are wired by the substrate builder from per-cell
# run_result + the logprob backfill).
COARSE_ZOO_FIELDS: tuple[str, ...] = (
    "cosine_l20_baseline",
    "cosine_response_headline",
    "source_base_rate",
    "base_rate_diff_neg_abs",
    "source_resp_len_mean",
    "bystander_resp_len_mean",
    "resp_len_diff_abs",
    "cosine_response_l7",
    "cosine_response_l14",
    "cosine_response_l21",
    "cosine_response_l27",
    "JS_sym_nats",
    "JS_from_source_nats",
    "JS_from_bystander_nats",
    "M_js",
    "KL_src_to_bys_nats",
    "KL_bys_to_src_nats",
    "KL_sym_nats",
)


# Per-cell field requirements for the cosine sweep JSON produced by
# ``scripts/issue404_predictor_cossim.py``. ``compute_coarse_zoo_for_arm``
# checks presence cell-by-cell and raises ``RuntimeError`` (naming the
# missing field + cell) if any is absent — mirroring
# ``load_existing_predictor_comparison`` at L148-155. The asymmetry that
# Codex round-5 finding #2 flagged (silent ``c.get(field, 0.0)`` in
# ``compute_coarse_zoo_for_arm`` vs ``raise RuntimeError`` in the syco
# loader) is closed here.
_REQUIRED_COSINE_FIELDS: tuple[str, ...] = (
    "cosine_response_l7",
    "cosine_response_l14",
    "cosine_response_l21",
    "cosine_response_l27",
)

# Per-cell field requirements for the JS/KL sweep JSON produced by
# ``scripts/issue458_predictor_jsdiv.py``. Same fail-loud contract as the
# cosine sweep.
_REQUIRED_JS_KL_FIELDS: tuple[str, ...] = (
    "JS_sym_nats",
    "JS_from_source_nats",
    "JS_from_bystander_nats",
    "KL_src_to_bys_nats",
    "KL_bys_to_src_nats",
    "KL_sym_nats",
)


def load_existing_predictor_comparison(
    *,
    arm: str,
    path: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """Load a pre-existing #480-style predictor_comparison.json from disk.

    Used for the syco arm where #480's per-cell sweep is already on disk.
    Returns a map ``{(source, bystander): {field: value}}`` for the 17
    coarse-zoo fields.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        RuntimeError: if any cell is missing a required field (fail-fast;
            the substrate builder must NOT silently emit partial cells).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Existing predictor_comparison.json missing: {path}. For the "
            f"{arm} arm, the production substrate requires a pre-existing "
            f"sweep on disk; re-run #480's predictor pipeline or pass "
            f"--smoke for the deterministic stub."
        )
    payload = json.loads(path.read_text())
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError(
            f"{path} does not have a non-empty 'cells' list "
            f"(found {type(cells).__name__}); the file may be corrupted."
        )
    out: dict[tuple[str, str], dict[str, float]] = {}
    for c in cells:
        src = c.get("source")
        bys = c.get("bystander")
        if not src or not bys:
            raise RuntimeError(f"cell missing source/bystander: {c!r}")
        cell_coarse: dict[str, float] = {}
        for f in COARSE_ZOO_FIELDS:
            if f not in c:
                raise RuntimeError(
                    f"cell ({src!r}, {bys!r}) missing coarse-zoo field {f!r}; "
                    f"the existing predictor_comparison.json at {path} is "
                    "incomplete. Re-run #480's predictor pipeline to backfill."
                )
            cell_coarse[f] = float(c[f])
        out[(src, bys)] = cell_coarse
    log.info("Loaded %d coarse-zoo cells from %s", len(out), path)
    return out


def _aggregate_panel_metrics(
    panel_path: Path,
) -> tuple[float, float, int]:
    """Extract (mean_resp_len_chars, mean_resp_len_tokens, n_rollouts).

    Reads one ``sycophancy_eval_<bystander>.json`` panel JSON and rolls up
    the per-rollout completion lengths so the substrate builder can compute
    ``source_resp_len_mean`` / ``bystander_resp_len_mean`` /
    ``resp_len_diff_abs``. Token count is approximated as
    ``len(completion.split())`` (character-stable, no tokenizer dependency).
    """
    payload = json.loads(panel_path.read_text())
    char_lens: list[int] = []
    tok_lens: list[int] = []
    for rec in payload.get("completions", []):
        text = rec.get("completion", "") or ""
        char_lens.append(len(text))
        tok_lens.append(len(text.split()))
    n = len(char_lens)
    mean_chars = sum(char_lens) / n if n else 0.0
    mean_toks = sum(tok_lens) / n if n else 0.0
    return float(mean_chars), float(mean_toks), n


def _build_cosine_cells(
    *,
    arm: str,
    cosine_payload: dict[str, Any],
    layer_cosine_path: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """Build per-(source, bystander) cosine cells from the sweep JSON.

    Fail-loud on any missing required field — mirrors the syco loader's
    ``load_existing_predictor_comparison`` presence check at L148-155. The
    silent ``c.get(field, 0.0)`` pattern that was here in round 5 was the
    asymmetry Codex round-5 finding #2 flagged.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    for c in cosine_payload.get("cells", []):
        src_name = c.get("source")
        bys_name = c.get("bystander")
        if not src_name or not bys_name:
            raise RuntimeError(
                f"compute_coarse_zoo_for_arm({arm!r}): cosine sweep cell "
                f"missing 'source' or 'bystander' key; cell payload = {c!r}; "
                f"file = {layer_cosine_path}"
            )
        # ``cosine_l20_baseline`` accepts either name (legacy alias
        # ``cosine_l20`` from older predictor sweeps). At least one MUST
        # be present.
        if "cosine_l20_baseline" in c:
            cosine_l20_value = float(c["cosine_l20_baseline"])
        elif "cosine_l20" in c:
            cosine_l20_value = float(c["cosine_l20"])
        else:
            raise RuntimeError(
                f"compute_coarse_zoo_for_arm({arm!r}): cosine sweep cell "
                f"(source={src_name!r}, bystander={bys_name!r}) missing both "
                f"'cosine_l20_baseline' and its legacy alias 'cosine_l20'. "
                f"Re-run scripts/issue404_predictor_cossim.py against this "
                f"arm's (source, bystander) panel; see {layer_cosine_path}."
            )
        # Per-layer cosines (l7/l14/l21/l27) are required by name — no
        # silent zero defaults. Mirrors the syco loader's
        # ``load_existing_predictor_comparison`` presence check at L148-155.
        for f in _REQUIRED_COSINE_FIELDS:
            if f not in c:
                raise RuntimeError(
                    f"compute_coarse_zoo_for_arm({arm!r}): cosine sweep cell "
                    f"(source={src_name!r}, bystander={bys_name!r}) missing "
                    f"required field {f!r}. Run "
                    f"scripts/issue404_predictor_cossim.py against this "
                    f"arm's (source, bystander) panel and verify the output "
                    f"schema at {layer_cosine_path}."
                )
        # ``cosine_response_headline`` falls back to ``cosine_response_l21``
        # by definition (the headline layer is l21 per #404's selection);
        # this is a documented derivation, not a silent default.
        headline_value = float(c.get("cosine_response_headline", c["cosine_response_l21"]))
        out[(src_name, bys_name)] = {
            "cosine_l20_baseline": cosine_l20_value,
            "cosine_response_l7": float(c["cosine_response_l7"]),
            "cosine_response_l14": float(c["cosine_response_l14"]),
            "cosine_response_l21": float(c["cosine_response_l21"]),
            "cosine_response_l27": float(c["cosine_response_l27"]),
            "cosine_response_headline": headline_value,
        }
    return out


def _build_js_kl_cells(
    *,
    arm: str,
    js_kl_payload: dict[str, Any],
    js_kl_path: Path,
) -> dict[tuple[str, str], dict[str, float]]:
    """Build per-(source, bystander) JS/KL cells from the sweep JSON.

    Fail-loud on any missing required field — same contract as
    ``_build_cosine_cells``. ``M_js = 1.0 - JS_sym_nats`` is a documented
    derivation when the sweep does not pre-compute it.
    """
    out: dict[tuple[str, str], dict[str, float]] = {}
    for c in js_kl_payload.get("cells", []):
        src_name = c.get("source")
        bys_name = c.get("bystander")
        if not src_name or not bys_name:
            raise RuntimeError(
                f"compute_coarse_zoo_for_arm({arm!r}): JS/KL sweep cell "
                f"missing 'source' or 'bystander' key; cell payload = {c!r}; "
                f"file = {js_kl_path}"
            )
        for f in _REQUIRED_JS_KL_FIELDS:
            if f not in c:
                raise RuntimeError(
                    f"compute_coarse_zoo_for_arm({arm!r}): JS/KL sweep cell "
                    f"(source={src_name!r}, bystander={bys_name!r}) missing "
                    f"required field {f!r}. Run "
                    f"scripts/issue458_predictor_jsdiv.py against this arm's "
                    f"(source, bystander) panel and verify the output schema "
                    f"at {js_kl_path}."
                )
        m_js_value = float(c.get("M_js", 1.0 - float(c["JS_sym_nats"])))
        out[(src_name, bys_name)] = {
            "JS_sym_nats": float(c["JS_sym_nats"]),
            "JS_from_source_nats": float(c["JS_from_source_nats"]),
            "JS_from_bystander_nats": float(c["JS_from_bystander_nats"]),
            "M_js": m_js_value,
            "KL_src_to_bys_nats": float(c["KL_src_to_bys_nats"]),
            "KL_bys_to_src_nats": float(c["KL_bys_to_src_nats"]),
            "KL_sym_nats": float(c["KL_sym_nats"]),
        }
    return out


def _load_base_per_bystander(
    *,
    arm: str,
    src: str,
    seed: int,
    src_panel_dir: Path,
    base_panel_dir: Path,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Read the per-source judged_<arm>.json and return base-panel rates.

    Returns ``(source_base_rate, base_per_bystander)``:

      - ``source_base_rate`` is the unadapted base model's arm-rate when
        WEARING the source persona — the diagonal of the base panel
        (``base_per_bystander[src]['rate']``). ``judge_refusal_panel`` +
        ``judge_em_panel`` populate this via the base-pass diagonal (round-6
        fix dropped the diagonal-skip from the base pass).
      - ``base_per_bystander`` is the full mapping ``{bystander: {"rate":
        float, ...}}`` for every bystander on the source's panel — required
        for the per-cell ``bystander_base_rate`` lookup that feeds
        ``base_rate_diff_neg_abs = -|src_rate - bys_rate|`` (round-7 fix
        for #480 schema match; Codex round-6 must-fix #1).

    A missing judged file, missing diagonal entry, or NaN diagonal rate is
    a structural upstream failure; silent ``0.0`` here would collapse
    predictor variance across all cells under this source (Codex round-5
    finding #1).
    """
    judged = src_panel_dir / f"judged_{arm}.json"
    if not judged.exists():
        raise FileNotFoundError(
            f"compute_coarse_zoo_for_arm({arm!r}): judged summary missing: "
            f"{judged}. Run "
            f"src/explore_persona_space/experiments/issue_518/"
            f"judge_{arm}_panel.py for source={src!r} (seed={seed}) before "
            f"invoking the substrate build."
        )
    jpayload = json.loads(judged.read_text())
    base_per_bys = jpayload.get("base_per_bystander", {})
    if not isinstance(base_per_bys, dict):
        raise RuntimeError(
            f"compute_coarse_zoo_for_arm({arm!r}): {judged} has malformed "
            f"'base_per_bystander' (expected dict, got "
            f"{type(base_per_bys).__name__})."
        )
    src_self = base_per_bys.get(src)
    if not isinstance(src_self, dict) or "rate" not in src_self:
        raise RuntimeError(
            f"compute_coarse_zoo_for_arm({arm!r}): {judged} is missing the "
            f"base-panel diagonal entry `base_per_bystander[{src!r}]['rate']`. "
            f"This is the source-self base rate; without it `source_base_rate` "
            f"collapses to a zero-variance predictor. Re-run "
            f"src/explore_persona_space/experiments/issue_518/"
            f"judge_{arm}_panel.py against this source (the round-6 fix "
            f"restored the base-panel diagonal); existing keys = "
            f"{sorted(base_per_bys.keys())[:5]!r}..."
        )
    source_base_rate = float(src_self["rate"])
    if source_base_rate != source_base_rate:  # NaN check
        raise RuntimeError(
            f"compute_coarse_zoo_for_arm({arm!r}): "
            f"`base_per_bystander[{src!r}]['rate']` in {judged} is NaN. For "
            f"the em arm this means no rollouts cleared the coherence filter "
            f"at the source-self diagonal; for refusal it means no judged "
            f"rollouts at all. Either way `source_base_rate` would be NaN "
            f"across every cell under this source. Inspect the base-panel "
            f"completions at "
            f"`{base_panel_dir}/sycophancy_eval_{src}.json` and re-run the "
            f"relevant judge."
        )
    return source_base_rate, base_per_bys


def _bystander_base_rate(
    *,
    arm: str,
    src: str,
    bys: str,
    base_per_bys: dict[str, dict[str, Any]],
    judged_path_for_msg: Path,
) -> float:
    """Look up ``base_per_bys[bys]['rate']`` (the bystander persona's own base
    rate on the source's panel) with the same fail-loud discipline as
    ``_load_base_per_bystander``. Required for the #480-schema-match
    ``base_rate_diff_neg_abs = -|src_rate - bys_rate|`` (round-7 fix)."""
    entry = base_per_bys.get(bys)
    if not isinstance(entry, dict) or "rate" not in entry:
        raise RuntimeError(
            f"compute_coarse_zoo_for_arm({arm!r}): bystander base-rate entry "
            f"`base_per_bystander[{bys!r}]['rate']` missing in "
            f"{judged_path_for_msg} for source={src!r}. Without it the #480-"
            f"schema `base_rate_diff_neg_abs = -|src_rate - bys_rate|` "
            f"silently degrades to `-|src_rate|`, breaking cross-arm "
            f"comparability with the syco arm. Re-run the judge against this "
            f"source so every bystander on its panel emits a base-pass rate. "
            f"Existing keys = {sorted(base_per_bys.keys())[:5]!r}..."
        )
    rate = float(entry["rate"])
    if rate != rate:  # NaN check
        raise RuntimeError(
            f"compute_coarse_zoo_for_arm({arm!r}): "
            f"`base_per_bystander[{bys!r}]['rate']` in {judged_path_for_msg} "
            f"is NaN. For em that means no rollouts cleared the coherence "
            f"filter on this bystander's base pass; for refusal it means no "
            f"judged base rollouts at all. The cell's "
            f"`base_rate_diff_neg_abs` would be NaN. Inspect the base-panel "
            f"completions and re-run the relevant judge."
        )
    return rate


def compute_coarse_zoo_for_arm(
    *,
    arm: str,
    slab_root: Path,
    judged_path: Path | None = None,
    base_judged_path: Path | None = None,
    seed: int = 42,
    layer_cosine_path: Path | None = None,
    js_kl_path: Path | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Compute the 17 coarse-zoo fields per (source, bystander) for one arm.

    Args:
        arm: 'refusal' or 'em'.
        slab_root: per-arm eval slab root (``eval_results/issue_518/<arm>/slab``).
        judged_path: optional override for the per-source judged_<arm>.json
            (default: derived per source).
        base_judged_path: optional override for the base-panel judged_<arm>.json
            (default: derived per arm).
        seed: training seed (default 42).
        layer_cosine_path: optional pre-computed cosine sweep JSON. If
            provided, cosine fields are loaded from it. If absent, this
            function raises (cosine MUST be computed pod-side via
            ``scripts/issue404_predictor_cossim.py`` before the substrate
            build).
        js_kl_path: optional pre-computed JS/KL sweep JSON. Same contract.

    Returns:
        Map ``{(source, bystander): {field: value}}`` for the 17 fields.

    Raises:
        FileNotFoundError: if any of the input files needed to compute a
            field is missing.
        RuntimeError: if a (source, bystander) cell has insufficient data.

    Note: This function does NOT spawn the predictor scripts itself —
    that's the production driver's job (the cosine + JS/KL sweeps run
    ONCE per arm before this function is called). This loader is a
    pure-read aggregator; the heavy lift is upstream.
    """
    if arm not in ("refusal", "em"):
        raise ValueError(f"Unsupported arm {arm!r}; expected 'refusal' or 'em'.")

    if layer_cosine_path is None or not layer_cosine_path.exists():
        raise FileNotFoundError(
            f"Cosine sweep JSON missing for arm {arm!r}: {layer_cosine_path}. "
            "Run scripts/issue404_predictor_cossim.py against the "
            f"{arm} arm's (source, bystander) panel before invoking the "
            "production substrate build."
        )
    if js_kl_path is None or not js_kl_path.exists():
        raise FileNotFoundError(
            f"JS/KL sweep JSON missing for arm {arm!r}: {js_kl_path}. "
            "Run scripts/issue458_predictor_jsdiv.py against the "
            f"{arm} arm's (source, bystander) panel before invoking the "
            "production substrate build."
        )

    cosine_payload = json.loads(layer_cosine_path.read_text())
    js_kl_payload = json.loads(js_kl_path.read_text())

    cosine_cells = _build_cosine_cells(
        arm=arm,
        cosine_payload=cosine_payload,
        layer_cosine_path=layer_cosine_path,
    )
    js_kl_cells = _build_js_kl_cells(
        arm=arm,
        js_kl_payload=js_kl_payload,
        js_kl_path=js_kl_path,
    )

    keys = sorted(set(cosine_cells) & set(js_kl_cells))
    if not keys:
        raise RuntimeError(
            f"No overlapping (source, bystander) keys between cosine "
            f"({layer_cosine_path}) and JS/KL ({js_kl_path}) sweeps for arm "
            f"{arm!r}."
        )

    # Response-length and base-rate aggregations need the per-source slab
    # JSONs. We compute on-demand per key.
    out: dict[tuple[str, str], dict[str, float]] = {}
    for src, bys in keys:
        src_panel_dir = slab_root / src / f"seed_{seed}"
        base_panel_dir = slab_root / "base" / f"seed_{seed}"
        src_panel = src_panel_dir / f"sycophancy_eval_{bys}.json"
        bys_panel = base_panel_dir / f"sycophancy_eval_{bys}.json"
        # Panel files are produced by ``sycophancy_implantation_411.
        # eval_one_source`` — required for response-length proxies. Missing
        # files are a real upstream failure, not a silent 0.0 case.
        if not src_panel.exists():
            raise FileNotFoundError(
                f"compute_coarse_zoo_for_arm({arm!r}): trained-side panel "
                f"file missing: {src_panel}. Re-run eval_one_source for "
                f"source={src!r} (seed={seed}) before invoking the substrate."
            )
        if not bys_panel.exists():
            raise FileNotFoundError(
                f"compute_coarse_zoo_for_arm({arm!r}): base-side panel file "
                f"missing: {bys_panel}. Re-run eval_one_source with "
                f"--hub-model-id Qwen/Qwen2.5-7B-Instruct (base, seed={seed}) "
                f"before invoking the substrate."
            )
        src_mean_chars, _, _ = _aggregate_panel_metrics(src_panel)
        bys_mean_chars, _, _ = _aggregate_panel_metrics(bys_panel)

        source_base_rate, base_per_bys = _load_base_per_bystander(
            arm=arm,
            src=src,
            seed=seed,
            src_panel_dir=src_panel_dir,
            base_panel_dir=base_panel_dir,
        )
        bystander_base_rate = _bystander_base_rate(
            arm=arm,
            src=src,
            bys=bys,
            base_per_bys=base_per_bys,
            judged_path_for_msg=src_panel_dir / f"judged_{arm}.json",
        )

        cell: dict[str, float] = {
            **cosine_cells[(src, bys)],
            **js_kl_cells[(src, bys)],
            "source_base_rate": source_base_rate,
            # Round-7 fix: match #480 schema in
            # ``eval_results/issue_480/_inputs/predictor_comparison.json``
            # (``base_rate_diff_neg_abs = -|src_rate - bys_rate|``). Prior
            # rounds emitted ``-|src_rate|``, which produced a different
            # quantity from the syco arm under the same field name and
            # silently corrupted the cross-arm
            # ``min(|rho_syco|, |rho_refusal|, |rho_em|)`` (Codex round-6
            # must-fix #1; reconciler verdict upheld 2026-06-08).
            "base_rate_diff_neg_abs": -abs(source_base_rate - bystander_base_rate),
            "source_resp_len_mean": src_mean_chars,
            "bystander_resp_len_mean": bys_mean_chars,
            "resp_len_diff_abs": abs(src_mean_chars - bys_mean_chars),
        }
        out[(src, bys)] = cell

    log.info(
        "Computed %d coarse-zoo cells for arm %r from cosine=%s + js_kl=%s",
        len(out),
        arm,
        layer_cosine_path,
        js_kl_path,
    )
    return out


def load_coarse_zoo_for_arm(
    *,
    arm: str,
    syco_predictor_comparison_path: Path | None = None,
    slab_root: Path | None = None,
    layer_cosine_path: Path | None = None,
    js_kl_path: Path | None = None,
    seed: int = 42,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Top-level dispatcher: syco -> load from disk, refusal/em -> compute.

    See module docstring for the two paths.
    """
    if arm == "syco":
        if syco_predictor_comparison_path is None:
            raise ValueError(
                "syco arm load requires syco_predictor_comparison_path "
                "(usually eval_results/issue_480/_inputs/predictor_comparison.json)."
            )
        return load_existing_predictor_comparison(arm=arm, path=syco_predictor_comparison_path)
    if arm in ("refusal", "em"):
        if slab_root is None:
            raise ValueError(
                f"{arm} arm load requires slab_root (usually eval_results/issue_518/{arm}/slab)."
            )
        return compute_coarse_zoo_for_arm(
            arm=arm,
            slab_root=slab_root,
            seed=seed,
            layer_cosine_path=layer_cosine_path,
            js_kl_path=js_kl_path,
        )
    raise ValueError(f"Unsupported arm {arm!r}; expected 'syco' | 'refusal' | 'em'.")
