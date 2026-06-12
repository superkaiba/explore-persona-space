#!/usr/bin/env python3
"""Verify that all experiment artifacts have been uploaded to permanent storage.

Called by the upload-verifier agent during status:uploading. Returns a JSON
report with PASS/FAIL per artifact category and permanent URLs for each.

Usage:
    # Check all artifacts for an issue
    uv run python scripts/verify_uploads.py --issue 42

    # Check with explicit artifact hints (from epm:results marker)
    uv run python scripts/verify_uploads.py \
        --issue 42 \
        --wandb-run "superkaiba/explore-persona-space/runs/abc123" \
        --hf-model "superkaiba1/explore-persona-space/issue-42-seed-42" \
        --pod pod3

    # HEAD-verify every HF / WandB URL claimed in the epm:results marker text
    # AND the body's ## Reproducibility section (phantom-URL detection — every
    # cited URL must actually resolve at its cited revision, not just be a
    # string in a sentinel). Required for training experiments per #456.
    uv run python scripts/verify_uploads.py --issue 42 \
        --claimed-urls-file /tmp/issue-42-claimed-urls.txt

    # Just check and print, no exit code (for interactive use)
    uv run python scripts/verify_uploads.py --issue 42 --no-fail

Sweep tasks (#608): when --wandb-run / --hf-model are omitted because the
run has no SINGLE path (per-cell adapters + per-cell WandB runs), the
training rows fall back to the task's epm:results reproducibility card
(``reproducibility_card``, or its ``reproducibility`` alias) — every
``adapter_paths`` entry is verified under ``hf_model_repo`` via
list_repo_files, and ``wandb_run_names`` + ``wandb_project`` resolve
per-cell runs by display name. When ``wandb_run_names`` is declared
WITHOUT ``wandb_project`` (#601: HF Trainer defaults the project to
``huggingface`` when WANDB_PROJECT is unset), the default entity's
projects are scanned — ``huggingface`` first — instead of hard-MISSING.
When NO wandb_* field is declared at all, the conventional per-issue
project (``<default_entity>/issue<N>``, runs named ``issue<N>_*``) is
probed before hard-MISSING (#608 follow-up); probe failures fail soft
back to MISSING. Explicit declarations always win unchanged.

Multi-launch runs post MULTIPLE epm:results markers (#601): a resume-pass
sentinel whose cells all ``resumed_skip`` carries an empty card
(``adapter_paths: {}``) that must not shadow the first marker's full
declaration, so the card is MERGED across all epm:results markers —
newest-wins per field, where an empty dict/list/string does not count as
a declaration (see ``merged_results_card``).

GCP-lane driver sentinels (#599) carry no reproducibility card at all —
per-seed provenance lives under ``production_provenance`` (e.g.
``production_provenance.seed42.hf_adapter_subfolder``). When a payload
declares no explicit card, an equivalent card is synthesized from those
keys plus any top-level wandb_* / hf_model_repo hints
(``_card_from_provenance``) so the hf_model / wandb_run rows stop
false-MISSing on artifacts that exist; explicit cards always win.

Claimed-URL repo types (#599): a claim citing a dataset repo WITHOUT the
``datasets/`` prefix (``hf://superkaiba1/explore-persona-space-data-private/...``)
used to resolve via the MODELS endpoint, 404, and turn the whole
claimed_urls row into ERROR. Bare ``org/repo`` claims are now probed for
their actual repo type (dataset-first for ``-data`` / ``-data-private``
repo-name suffixes) and rewritten to the ``datasets/`` form before
HEAD-checking (``resolve_claimed_repo_types``); a claim resolving as
NEITHER type is reported claimed-but-absent (FAIL) without aborting the
rest of the scan.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Make the repo's src/ importable so we can reuse the canonical HF/WandB
# HEAD-check helper (verify_artifacts_exist) instead of reimplementing it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Repos
HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Map task-workflow frontmatter ``kind`` values to the experiment type whose
# checklist rows apply when the caller omits --type. ``experiment`` stays
# "training" (the conservative default): frontmatter cannot distinguish a
# training run from an eval-only one, and silently relaxing the HF-model /
# WandB-run rows for a task that DID train would weaken the Step 8 hard
# gate. Callers that know better (the upload-verifier receives the
# experiment type as an input) pass --type explicitly (#563).
_KIND_TO_EXPERIMENT_TYPE = {
    "experiment": "training",
    "analysis": "analysis",
    "infra": "analysis",
    "batch": "analysis",
    "survey": "analysis",
}


def infer_experiment_type(issue_num: int) -> tuple[str, str]:
    """Infer the experiment type from the task's frontmatter ``kind``.

    Returns ``(experiment_type, source)``: source is ``frontmatter-kind``
    when the task's ``kind`` mapped cleanly, ``default`` when the task /
    frontmatter could not be read or the kind is unknown. Failures fall
    back to ``training`` — the STRICTEST type — so a broken inference can
    only over-demand rows, never silently relax the gate.
    """
    try:
        from explore_persona_space.task_workflow import get_task

        kind = str(get_task(issue_num)["frontmatter"].get("kind", "")).strip()
    except Exception as e:
        logger.warning(
            "could not read task %s frontmatter (%s); assuming experiment_type=training",
            issue_num,
            e,
        )
        return "training", "default"
    if kind in _KIND_TO_EXPERIMENT_TYPE:
        return _KIND_TO_EXPERIMENT_TYPE[kind], "frontmatter-kind"
    logger.warning("unknown kind %r on task %s; assuming experiment_type=training", kind, issue_num)
    return "training", "default"


def check_hf_hub_path(
    repo_id: str,
    path_in_repo: str,
    repo_type: str = "model",
    revision: str | None = None,
) -> dict:
    """Check if a path exists on HF Hub at the given revision.

    ``revision`` defaults to ``main``. Pass a commit SHA to HEAD-verify that
    the files actually exist at the pinned revision a downstream consumer
    will dereference — this is what the phantom-URL gate needs (a string
    claiming ``/tree/<sha>/...`` is not the same as the files being there).
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)
        prefix = path_in_repo.rstrip("/") + "/"
        matching = [f for f in files if f.startswith(prefix) or f == path_in_repo]
        rev_url = revision or "main"
        if matching:
            url = f"https://huggingface.co/{repo_id}/tree/{rev_url}/{path_in_repo}"
            return {"status": "OK", "url": url, "file_count": len(matching)}
        return {
            "status": "MISSING",
            "url": "",
            "detail": f"No files under {path_in_repo} at revision {rev_url}",
        }
    except Exception as e:
        return {"status": "ERROR", "url": "", "detail": str(e)}


# Claimed-text blobs are frequently JSON (epm:results sentinels are JSON), so
# every URL is immediately followed by '",' (or '\",' when the JSON is nested).
# hub.py's _HF_URL_RE revision/path character classes exclude only '/',
# whitespace, ')' and ']', so that trailing punctuation rides into the probed
# revision/path and every HEAD check misses — a false claimed_urls FAIL
# (incident #541, 2026-06-10). Extract URL candidates permissively, strip
# trailing punctuation, and hand verify_artifacts_exist a sanitized
# one-URL-per-line view it parses cleanly.
_CLAIMED_URL_RE = re.compile(r"(?:https?|hf)://\S+")
# NOTE: '.' is deliberately NOT stripped — artifact paths legitimately end in
# '.json' / '.safetensors'; a sentence-final period stays a (pre-existing,
# rare) false MISS rather than risking real-suffix truncation.
_TRAILING_PUNCT = "\\'\",;)]}>`"


def _strip_trailing_punct(url: str) -> str:
    """Strip trailing JSON/markdown punctuation from a URL candidate.

    A trailing ``.`` is removed ONLY when the character beneath it is itself
    in the punctuation set (the markdown sentence-end case, e.g. ``` `url`. ```)
    — a period directly after a path character is kept so real suffixes like
    ``.json`` / ``.safetensors`` never truncate.
    """
    while url and (
        url[-1] in _TRAILING_PUNCT
        or (url[-1] == "." and len(url) >= 2 and url[-2] in _TRAILING_PUNCT)
    ):
        url = url[:-1].rstrip(".")
    return url


def extract_claimed_urls(text: str) -> list[str]:
    """Extract HF/WandB/hf:// URL candidates from a claimed-text blob.

    Strips trailing JSON/markdown punctuation (quotes, commas, semicolons,
    closing brackets/braces/parens, backticks, backslashes) from each match
    and de-duplicates preserving first-seen order. Returns the cleaned URLs.
    """
    return list(dict.fromkeys(_strip_trailing_punct(u) for u in _CLAIMED_URL_RE.findall(text)))


# ── claimed-URL repo-type resolution (dataset-repo fallback, #599) ─────────────
# Bare ``org/repo`` HF claims default to repo_type="model" downstream
# (hub.py's _kind_to_repo_type), so a dataset repo cited without the
# ``datasets/`` prefix 404s on the MODELS endpoint and the propagated
# RepositoryNotFoundError turned the WHOLE claimed_urls row into ERROR
# (#599: ``hf://superkaiba1/explore-persona-space-data-private/...``).
# Probe each bare claim's actual repo type and rewrite dataset claims to
# the prefixed form verify_artifacts_exist resolves correctly.

_BARE_HF_CLAIM_RE = re.compile(
    r"^(?P<scheme>https?://huggingface\.co/|hf://)"
    r"(?!datasets/|spaces/)"
    r"(?P<repo>[\w.\-]+/[\w.\-]+)"
    r"(?P<rest>(?:[/@].*)?)$"
)

# Repo-name suffixes that are dataset repos by project convention — probe
# the dataset endpoint FIRST so the common case costs one repo_info call.
_DATASET_FIRST_SUFFIXES = ("-data", "-data-private")


def _hf_repo_type_for(api, repo_id: str, cache: dict) -> str | None:
    """Resolve whether a bare repo id is a model or a dataset repo (cached).

    Returns ``"model"`` / ``"dataset"``, or ``None`` when the repo resolves
    as NEITHER (``RepositoryNotFoundError`` on both endpoints — a phantom
    claim, or a private repo the ambient HF_TOKEN cannot see). Non-404
    errors propagate so a transient outage is not misread as "missing".
    """
    if repo_id in cache:
        return cache[repo_id]
    from huggingface_hub.utils import RepositoryNotFoundError

    name = repo_id.split("/", 1)[-1]
    order = ("dataset", "model") if name.endswith(_DATASET_FIRST_SUFFIXES) else ("model", "dataset")
    resolved: str | None = None
    for repo_type in order:
        try:
            api.repo_info(repo_id, repo_type=repo_type)
            resolved = repo_type
            break
        except RepositoryNotFoundError:
            continue
    cache[repo_id] = resolved
    return resolved


def resolve_claimed_repo_types(urls: list[str]) -> tuple[list[str], dict[str, str], list[str]]:
    """Qualify bare HF repo claims with their actual repo type (#599).

    Each claim matching ``_BARE_HF_CLAIM_RE`` (an HF URL whose repo id is
    NOT already ``datasets/`` / ``spaces/``-prefixed) is probed via
    ``repo_info`` (one call per unique repo). Dataset-repo claims are
    rewritten to the ``datasets/``-prefixed form so the downstream
    existence check hits the right endpoint; model claims pass through
    unchanged; claims resolving as neither type are split out as phantoms
    so ONE bad repo claim no longer aborts the whole scan into ERROR.

    Returns ``(resolved_urls, rewritten_to_original, phantom_urls)``:
    ``resolved_urls`` feed ``verify_artifacts_exist`` (dataset claims
    rewritten), ``rewritten_to_original`` maps rewritten → as-cited so
    reports name the URL the way the task cited it, and ``phantom_urls``
    are reported claimed-but-absent (FAIL, not ERROR).
    """
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    cache: dict = {}
    resolved_urls: list[str] = []
    rewritten_to_original: dict[str, str] = {}
    phantoms: list[str] = []
    for url in urls:
        m = _BARE_HF_CLAIM_RE.match(url)
        if not m:
            resolved_urls.append(url)
            continue
        repo_type = _hf_repo_type_for(api, m.group("repo"), cache)
        if repo_type == "dataset":
            rewritten = f"{m.group('scheme')}datasets/{m.group('repo')}{m.group('rest')}"
            resolved_urls.append(rewritten)
            rewritten_to_original[rewritten] = url
        elif repo_type is None:
            phantoms.append(url)
        else:
            resolved_urls.append(url)
    return resolved_urls, rewritten_to_original, phantoms


def check_claimed_urls_resolve(claimed_text_path: str | Path) -> dict:
    """HEAD-verify every HF/WandB URL claimed in a text blob actually resolves.

    The blob is typically the concatenation of the ``epm:results`` marker
    text + the body's ``## Reproducibility`` section. URLs are first
    extracted and stripped of trailing JSON/markdown punctuation (see
    ``extract_claimed_urls``), bare ``org/repo`` HF claims are qualified
    with their actual repo type — a dataset repo cited without the
    ``datasets/`` prefix is rewritten rather than 404ing on the MODELS
    endpoint (#599; see ``resolve_claimed_repo_types``) — then
    existence-checked via
    ``explore_persona_space.orchestrate.hub.verify_artifacts_exist`` (the
    same helper /issue Step 6a.5 uses pre-launch to block on phantom
    carry-over artifacts) so behavior stays consistent at both gates.

    A claimed-but-absent URL is a hard ``FAIL`` — that is exactly the
    phantom-checkpoint condition that lets a write-up cite a file nothing
    ever uploaded. Use this BEFORE PASSing upload-verification.

    Args:
        claimed_text_path: Path to a UTF-8 text file containing the
            epm:results marker body + the Reproducibility section (and
            anything else cited). The helper scans for HF / WandB URLs;
            non-URL text is ignored.

    Returns:
        A status dict shaped like other ``check_*`` helpers.
        ``status == "OK"`` means every URL scanned resolved; ``"FAIL"``
        means one or more URLs were strings without a real artifact;
        ``"SKIP"`` means no URLs were scanned (e.g. caller did not pass
        a file); ``"ERROR"`` means a transport / auth issue propagated.
    """
    if not claimed_text_path:
        return {
            "status": "SKIP",
            "url": "",
            "detail": "No --claimed-urls-file provided",
        }
    claimed_text_path = Path(claimed_text_path)
    if not claimed_text_path.exists() or not claimed_text_path.is_file():
        return {
            "status": "ERROR",
            "url": "",
            "detail": f"claimed-urls file missing or not a file: {claimed_text_path}",
        }
    try:
        from explore_persona_space.orchestrate.hub import verify_artifacts_exist

        urls = extract_claimed_urls(claimed_text_path.read_text(encoding="utf-8"))
        # Qualify bare org/repo claims with their actual repo type (#599):
        # dataset claims get the datasets/ prefix the downstream checker
        # needs; claims whose repo resolves as neither type become
        # deterministic phantoms instead of aborting the scan with ERROR.
        urls, rewritten_to_original, phantoms = resolve_claimed_repo_types(urls)
        # Write the sanitized one-URL-per-line view to a temp file:
        # verify_artifacts_exist takes a path and runs its own URL regexes,
        # which terminate cleanly at end-of-line once trailing punctuation
        # has been stripped here.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".claimed-urls.txt", delete=False
        ) as tf:
            tf.write("\n".join(urls) + ("\n" if urls else ""))
            sanitized_path = Path(tf.name)
        try:
            ok, missing = verify_artifacts_exist(sanitized_path)
        finally:
            sanitized_path.unlink(missing_ok=True)
        if ok and not phantoms:
            detail = "every claimed HF/WandB URL resolves at its cited revision"
            if rewritten_to_original:
                detail += (
                    f"; {len(rewritten_to_original)} bare dataset-repo claim(s) "
                    "resolved via repo_type=dataset (#599)"
                )
            return {"status": "OK", "url": str(claimed_text_path), "detail": detail}
        # Report missing URLs the way the task cited them (un-rewritten).
        missing_cited = phantoms + [rewritten_to_original.get(u, u) for u in missing]
        detail = "claimed-but-absent URLs (phantom): " + "; ".join(missing_cited)
        if phantoms:
            detail += (
                " [repo resolves as neither model nor dataset — phantom repo, "
                "or private without HF_TOKEN access]"
            )
        return {"status": "FAIL", "url": "", "detail": detail}
    except Exception as e:
        return {"status": "ERROR", "url": "", "detail": str(e)}


def check_wandb_run(run_path: str) -> dict:
    """Check if a WandB run exists and is accessible."""
    try:
        import wandb

        api = wandb.Api()
        run = api.run(run_path)
        url = run.url
        return {"status": "OK", "url": url, "state": run.state}
    except Exception as e:
        return {"status": "MISSING", "url": "", "detail": str(e)}


def check_wandb_artifact(artifact_path: str) -> dict:
    """Check if a WandB artifact exists."""
    try:
        import wandb

        api = wandb.Api()
        artifact = api.artifact(artifact_path)
        url = f"https://wandb.ai/{artifact.entity}/{artifact.project}/artifacts/{artifact.type}/{artifact.name}"
        return {"status": "OK", "url": url, "size": artifact.size}
    except Exception as e:
        return {"status": "MISSING", "url": "", "detail": str(e)}


# ── epm:results reproducibility-card fallback (#608, #601) ────────────────────
# Multi-cell sweeps declare their artifacts per cell (an ``adapter_paths``
# dict + per-cell WandB run names) inside the epm:results payload's
# ``reproducibility_card`` (alias ``reproducibility``) — there is no single
# --hf-model / --wandb-run value to pass. Without this fallback every sweep
# task produced a false mechanical FAIL on the wandb_run / hf_model rows
# that the upload-verifier had to supersede row-by-row (same false-FAIL
# class as incident #563). Multi-launch runs post several epm:results
# markers, and a resume-pass sentinel can carry an EMPTY card (#601), so
# the card is merged across all markers, newest-wins per declared field.
# The fallback fires ONLY when the caller declared no single path; explicit
# declarations always win unchanged.


def _extract_first_json_object(text: str) -> dict | None:
    """Parse the first JSON object embedded in a marker note.

    epm:results notes are frequently prose-prefixed (e.g. the orchestrator's
    "[drained from pod sentinel ...]" line on #608) with the JSON payload
    after it, so scan ``{`` candidates left-to-right and return the first
    one that parses as a dict.
    """
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return obj
        idx = text.find("{", idx + 1)
    return None


# Producers name the card ``reproducibility_card`` (canonical, #608) or
# ``reproducibility`` (the #601 sweep dispatcher); canonical key wins when
# both are present in one payload.
_CARD_KEYS = ("reproducibility_card", "reproducibility")

# Top-level payload keys a GCP-lane driver sentinel may carry alongside
# ``production_provenance`` (#599) — copied into the synthesized card so a
# declared wandb project / model-repo hint is not lost.
_PROVENANCE_HINT_KEYS = (
    "hf_model_repo",
    "hf_model_path",
    "wandb_run_path",
    "wandb_run",
    "wandb_run_names",
    "wandb_project",
    "wandb_entity",
)


def _card_from_provenance(payload: dict) -> dict | None:
    """Synthesize a reproducibility card from a GCP-lane driver sentinel (#599).

    ``epm:results`` sentinels written by GCP-lane drivers declare per-seed
    adapters as ``production_provenance.<cell>.hf_adapter_subfolder``
    (optionally ``.wandb_run_name``) instead of a ``reproducibility_card``,
    so the card fallback false-MISSed the hf_model / wandb_run rows even
    when every artifact existed. Additive: consulted ONLY when the payload
    carries no explicit card (``_card_from_payload`` tries ``_CARD_KEYS``
    first). Top-level wandb_* / hf_model_repo hints are carried over.
    Returns ``None`` when ``production_provenance`` declares nothing usable.
    """
    prov = payload.get("production_provenance")
    if not isinstance(prov, dict):
        return None
    adapter_paths: dict = {}
    run_names: dict = {}
    for cell, info in prov.items():
        if not isinstance(info, dict):
            continue
        subfolder = info.get("hf_adapter_subfolder")
        if _is_declared(subfolder):
            adapter_paths[str(cell)] = str(subfolder)
        run_name = info.get("wandb_run_name")
        if _is_declared(run_name):
            run_names[str(cell)] = str(run_name)
    card: dict = {}
    if adapter_paths:
        card["adapter_paths"] = adapter_paths
    if run_names:
        card["wandb_run_names"] = run_names
    for key in _PROVENANCE_HINT_KEYS:
        if key not in card and _is_declared(payload.get(key)):
            card[key] = payload[key]
    if not card:
        return None
    card["_card_provenance"] = (
        "synthesized from epm:results production_provenance (no reproducibility_card)"
    )
    return card


def _card_from_payload(payload: dict) -> dict | None:
    """Return the reproducibility card dict from a parsed epm:results payload.

    Explicit cards win; a GCP-lane sentinel with no card falls back to
    synthesis from ``production_provenance`` (#599, ``_card_from_provenance``).
    """
    for key in _CARD_KEYS:
        card = payload.get(key)
        if isinstance(card, dict):
            return card
    return _card_from_provenance(payload)


def _is_declared(value) -> bool:
    """True when a card field actually declares something (non-empty).

    A resume-pass re-post can carry the card SHAPE with empty contents
    (#601: ``adapter_paths: {}`` after every cell ``resumed_skip``) — an
    empty dict/list/string or None is not a declaration and must not
    shadow an earlier marker's real one.
    """
    return value is not None and value != "" and value != {} and value != []


def merged_results_card(events: list[dict]) -> dict | None:
    """Merge reproducibility cards across ALL ``epm:results`` events.

    Multi-launch runs legitimately post several ``epm:results`` markers
    (resume relaunches, drained sentinels), and a later sentinel can carry
    an empty card that would shadow the first marker's full declaration
    (#601: a resume pass with every cell ``resumed_skip`` posted
    ``adapter_paths: {}``, masking 16 verified adapter paths). Each FIELD
    therefore resolves newest-wins: the value comes from the newest card
    that declares it non-empty (``_is_declared``). When any field falls
    back past the newest card, the merged card carries a
    ``_card_provenance`` note that the row checks append to their detail.
    Returns ``None`` when no event declares a card (or every card is
    entirely empty) — the caller falls through to the strict MISSING row.
    """
    cards: list[tuple[dict, str]] = []  # newest first
    for ev in reversed(events):
        if str(ev.get("kind", "")) != "epm:results":
            continue
        payload = _extract_first_json_object(str(ev.get("note", "")))
        if payload is None:
            continue
        card = _card_from_payload(payload)
        if card is not None:
            cards.append((card, str(ev.get("ts", "")) or "unknown-ts"))
    if not cards:
        return None
    merged: dict = {}
    fallback_fields: dict[str, str] = {}
    for pos, (card, ts) in enumerate(cards):
        for key, value in card.items():
            if key in merged or not _is_declared(value):
                continue
            if key.startswith("_") and pos > 0:
                # Provenance notes (e.g. a synthesized card's
                # ``_card_provenance`` — #599) travel only with the newest
                # card; an older card's note would misattribute the merged
                # fields.
                continue
            merged[key] = value
            if pos > 0:
                fallback_fields[key] = ts
    if fallback_fields:
        note = "field(s) declared by an earlier epm:results marker, not the latest: " + ", ".join(
            f"{k} @ {ts}" for k, ts in sorted(fallback_fields.items())
        )
        existing = merged.get("_card_provenance")
        merged["_card_provenance"] = f"{existing}; {note}" if existing else note
    return merged or None


def _append_card_provenance(result: dict, card: dict) -> dict:
    """Append the cross-marker fallback note to a card-check result's detail."""
    provenance = card.get("_card_provenance")
    if provenance:
        detail = result.get("detail", "")
        result["detail"] = f"{detail} [{provenance}]".strip() if detail else f"[{provenance}]"
    return result


def _load_results_card(issue_num: int) -> dict | None:
    """Read the task's events and return its merged reproducibility card.

    Fail-soft: a missing task / unreadable events file returns ``None`` and
    the caller falls through to the strict MISSING row — a broken fallback
    can only over-demand, never silently relax the gate.
    """
    try:
        from explore_persona_space.task_workflow import list_events

        return merged_results_card(list_events(issue_num))
    except Exception as e:
        logger.warning("could not read epm:results card for task %s (%s)", issue_num, e)
        return None


# A `<arm>` / `<source>` / `<seed>`-style template placeholder inside a card
# field — the signature of the #612 prose-template shape.
_PLACEHOLDER_RE = re.compile(r"<[^<>\s][^<>]{0,40}>")


def _prose_declaration_row(field: str, value: str) -> dict:
    """MISSING row naming a prose-template card declaration (#612).

    Producers MUST declare ``adapter_paths`` / ``wandb_run_names`` as
    structured per-cell dicts/lists of REAL paths / run names — the
    epm:results sentinel contract, ``.claude/skills/issue/SKILL.md``
    Step 7. A prose summary string (e.g. ``adapters/issue_612/<arm>/
    <source>_seed<S> (16 adapters)``) resolves to nothing; silently
    ignoring it produced an uninformative generic MISSING on a
    fully-uploaded sweep that cost a manual investigation (#612). The
    row stays MISSING — this is diagnostic-only.
    """
    snippet = value if len(value) <= 100 else value[:97] + "..."
    placeholders = " with <...> template placeholders" if _PLACEHOLDER_RE.search(value) else ""
    return {
        "status": "MISSING",
        "url": "",
        "detail": (
            f"reproducibility_card declares {field} as a prose string{placeholders}, "
            "not a per-cell dict/list of real values — producer-contract violation "
            "(epm:results sentinel contract, .claude/skills/issue/SKILL.md Step 7; "
            f"incident #612): {snippet!r}"
        ),
        "source": "epm:results reproducibility_card",
    }


def check_hf_model_from_card(card: dict) -> dict | None:
    """Verify model paths declared in an epm:results reproducibility_card.

    Accepts a per-cell ``adapter_paths`` dict/list and/or a single
    ``hf_model_path``, all under ``hf_model_repo`` (default
    ``HF_MODEL_REPO``). Each path is existence-checked via
    ``check_hf_hub_path``. A STRING-valued ``adapter_paths`` (the #612
    prose-template shape) is unverifiable; instead of silently ignoring
    it (which read as a generic declaration-gap MISSING on a
    fully-uploaded sweep), the row names the producer-contract violation
    (``_prose_declaration_row``). Returns ``None`` when the card declares
    no model paths (caller falls through to the MISSING row).
    """
    repo = str(card.get("hf_model_repo") or HF_MODEL_REPO)
    paths: list[str] = []
    adapter_paths = card.get("adapter_paths")
    prose_violation: dict | None = None
    if isinstance(adapter_paths, dict):
        paths.extend(str(p) for p in adapter_paths.values())
    elif isinstance(adapter_paths, list):
        paths.extend(str(p) for p in adapter_paths)
    elif isinstance(adapter_paths, str) and _is_declared(adapter_paths):
        prose_violation = _prose_declaration_row("adapter_paths", adapter_paths)
    single = card.get("hf_model_path")
    if single:
        paths.append(str(single))
    paths = list(dict.fromkeys(paths))
    if not paths:
        if prose_violation is not None:
            return _append_card_provenance(prose_violation, card)
        return None

    absent: list[str] = []
    errored = False
    total_files = 0
    for p in paths:
        res = check_hf_hub_path(repo, p, "model")
        if res["status"] == "OK":
            total_files += res.get("file_count", 0)
        else:
            errored = errored or res["status"] == "ERROR"
            absent.append(f"{p} ({res.get('detail') or res['status']})")
    if absent:
        result = {
            "status": "ERROR" if errored else "MISSING",
            "url": "",
            "detail": (
                f"reproducibility_card declares {len(paths)} model path(s) under "
                f"{repo}; unresolved: " + "; ".join(absent[:5])
            ),
            "source": "epm:results reproducibility_card",
        }
    else:
        result = {
            "status": "OK",
            "url": f"https://huggingface.co/{repo}/tree/main",
            "file_count": total_files,
            "detail": (
                f"all {len(paths)} model path(s) from the epm:results "
                f"reproducibility_card resolve on {repo}"
            ),
            "source": "epm:results reproducibility_card",
        }
    if prose_violation is not None:
        # A real hf_model_path resolved (or failed) above, but the card ALSO
        # carried an unverifiable prose adapter_paths — keep that visible.
        result["detail"] = f"{result['detail']}; ALSO: {prose_violation['detail']}"
    return _append_card_provenance(result, card)


def check_wandb_runs_by_name(project_path: str, run_names: list[str]) -> dict:
    """Resolve per-cell WandB runs by display name within one project.

    ``project_path`` is ``entity/project`` (or bare ``project`` for the
    default entity). Every declared name must resolve for OK.
    """
    try:
        import wandb

        api = wandb.Api()
        runs = api.runs(project_path, filters={"displayName": {"$in": run_names}})
        found = {r.name for r in runs}
        missing = [n for n in run_names if n not in found]
        if missing:
            return {
                "status": "MISSING",
                "url": "",
                "detail": (
                    f"{len(missing)}/{len(run_names)} declared run name(s) not found "
                    f"in {project_path}: " + ", ".join(missing[:5])
                ),
            }
        return {
            "status": "OK",
            "url": f"https://wandb.ai/{project_path}",
            "detail": f"all {len(run_names)} declared run name(s) resolve in {project_path}",
        }
    except Exception as e:
        return {"status": "MISSING", "url": "", "detail": str(e)}


# HF Trainer defaults the WandB project to "huggingface" when WANDB_PROJECT
# is unset, so a sentinel that follows the common declared-names pattern but
# omits wandb_project usually has its runs there (#601: two runs existed in
# thomasjiralerspong/huggingface but the row hard-MISSed, forcing a manual
# override to PASS). Cap the project scan so a huge entity stays cheap.
_WANDB_DEFAULT_PROJECT_SCAN_CAP = 25


def check_wandb_runs_default_project(run_names: list[str], entity: str | None = None) -> dict:
    """Resolve declared run display names when the card omits ``wandb_project``.

    Scans the default entity's ``huggingface`` project first (the HF
    Trainer default when WANDB_PROJECT is unset — #601), then the entity's
    other projects (capped at ``_WANDB_DEFAULT_PROJECT_SCAN_CAP``), using
    the same server-side displayName filter as ``check_wandb_runs_by_name``
    so big projects are never paged client-side. OK requires every declared
    name to resolve within ONE project; the resolved project is reported in
    the detail instead of MISSING.
    """
    try:
        import wandb

        api = wandb.Api()
        entity = entity or api.default_entity
        if not entity:
            return {
                "status": "MISSING",
                "url": "",
                "detail": (
                    "card declares wandb_run_names without wandb_project and no "
                    "default WandB entity is configured to scan"
                ),
            }
        project_names = ["huggingface"]
        for proj in api.projects(entity):
            if proj.name not in project_names:
                project_names.append(proj.name)
            if len(project_names) >= _WANDB_DEFAULT_PROJECT_SCAN_CAP:
                break
        best_partial: tuple[int, str] | None = None
        probe_error: str | None = None
        for project in project_names:
            try:
                runs = api.runs(f"{entity}/{project}", filters={"displayName": {"$in": run_names}})
                found = {r.name for r in runs}
            except Exception as e:
                # The "huggingface" project may not exist for this entity;
                # record the probe failure and keep scanning real projects.
                probe_error = f"{entity}/{project}: {e}"
                continue
            if all(n in found for n in run_names):
                return {
                    "status": "OK",
                    "url": f"https://wandb.ai/{entity}/{project}",
                    "detail": (
                        f"all {len(run_names)} declared run name(s) resolve in "
                        f"default-entity project {entity}/{project} (card omitted "
                        "wandb_project; HF Trainer default-project fallback)"
                    ),
                }
            if found and (best_partial is None or len(found) > best_partial[0]):
                best_partial = (len(found), project)
        detail = (
            f"card declares {len(run_names)} wandb_run_names without wandb_project; "
            f"no single project under entity {entity} resolves all of them "
            f"(scanned {len(project_names)} project(s) starting with huggingface)"
        )
        if best_partial:
            detail += (
                f"; best partial: {best_partial[0]}/{len(run_names)} in {entity}/{best_partial[1]}"
            )
        if probe_error and not best_partial:
            detail += f"; last probe error: {probe_error}"
        return {"status": "MISSING", "url": "", "detail": detail}
    except Exception as e:
        return {"status": "MISSING", "url": "", "detail": str(e)}


def check_wandb_from_card(card: dict) -> dict | None:
    """Verify WandB runs declared in an epm:results reproducibility_card.

    Accepts a single ``wandb_run_path`` / ``wandb_run`` (delegates to
    ``check_wandb_run``) or per-cell ``wandb_run_names`` (dict or list) +
    ``wandb_project`` (optional ``wandb_entity``). When ``wandb_run_names``
    is declared WITHOUT ``wandb_project``, falls back to scanning the
    default entity's projects — ``huggingface`` first, the HF Trainer
    default when WANDB_PROJECT is unset (#601) — via
    ``check_wandb_runs_default_project``. Prose declarations (the #612
    template shape) are diagnosed instead of producing garbage rows: a
    ``wandb_run_path`` containing whitespace / ``<...>`` placeholders is
    never a real run path (the API call would only yield an opaque error
    string), and a STRING ``wandb_run_names`` would otherwise iterate
    into per-CHARACTER "run names" — both get the
    ``_prose_declaration_row`` contract-violation detail (row stays
    MISSING either way). Returns ``None`` when the card declares no
    WandB fields.
    """
    single = card.get("wandb_run_path") or card.get("wandb_run")
    if isinstance(single, str) and (
        _PLACEHOLDER_RE.search(single) or re.search(r"\s", single.strip())
    ):
        return _append_card_provenance(
            _prose_declaration_row("wandb_run_path/wandb_run", single), card
        )
    if single:
        result = check_wandb_run(str(single))
        result["source"] = "epm:results reproducibility_card"
        return _append_card_provenance(result, card)
    names = card.get("wandb_run_names")
    if isinstance(names, str) and _is_declared(names):
        return _append_card_provenance(_prose_declaration_row("wandb_run_names", names), card)
    if isinstance(names, dict):
        names = list(names.values())
    project = card.get("wandb_project")
    if names and project:
        entity = card.get("wandb_entity")
        project_path = f"{entity}/{project}" if entity else str(project)
        result = check_wandb_runs_by_name(project_path, [str(n) for n in names])
        result["source"] = "epm:results reproducibility_card"
        return _append_card_provenance(result, card)
    if names:
        # Declared names but NO project (#601): HF Trainer runs default to
        # project "huggingface" when WANDB_PROJECT is unset, so scan the
        # default entity's projects instead of hard-MISSING.
        result = check_wandb_runs_default_project(
            [str(n) for n in names], entity=card.get("wandb_entity")
        )
        result["source"] = "epm:results reproducibility_card"
        return _append_card_provenance(result, card)
    return None


def check_wandb_runs_convention_project(issue_num: int) -> dict | None:
    """Probe the conventional ``<default_entity>/issue<N>`` WandB project.

    Last-resort fallback when neither the CLI nor any epm:results
    reproducibility_card declares a wandb_* field: dispatchers
    conventionally log per-issue runs to the project ``issue<N>`` under
    the default entity, named ``issue<N>_*`` (#608 follow-up: all 12 runs
    resolved server-side at thomasjiralerspong/issue608 while the
    wandb_run row mechanically FAILed on the card's declaration gap).
    Returns an OK row carrying a declaration-gap note when at least one
    conventionally named run resolves there; returns ``None`` — keeping
    today's MISSING behavior — when no run matches OR the probe fails for
    any reason (fail-soft: a WandB API error must not change the row). A
    per-issue project only holds that issue's runs, so client-side name
    filtering is cheap (unlike the default-project scan, which must
    filter server-side).
    """
    prefix = f"issue{issue_num}_"
    try:
        import wandb

        api = wandb.Api()
        entity = api.default_entity
        if not entity:
            return None
        runs = api.runs(f"{entity}/issue{issue_num}")
        names = [str(r.name) for r in runs if str(r.name).startswith(prefix)]
    except Exception as e:
        logger.warning(
            "conventional WandB project probe failed for issue %s (%s); keeping MISSING",
            issue_num,
            e,
        )
        return None
    if not names:
        return None
    return {
        "status": "OK",
        "url": f"https://wandb.ai/{entity}/issue{issue_num}",
        "detail": (
            f"{len(names)} run(s) named {prefix}* resolve in conventional "
            f"project {entity}/issue{issue_num}; no reproducibility_card "
            "declares wandb_run_path / wandb_run_names (declaration gap — "
            "the results sentinel should declare them)"
        ),
        "source": "wandb project-naming convention (no card declaration)",
    }


def _issue_branch_ref(issue_num: int) -> str | None:
    """Return the first existing git ref for the issue branch, or None.

    Prefers the local worktree branch (``issue-<N>``) over the pushed
    remote-tracking ref (``origin/issue-<N>``). No fetch is performed —
    only refs already known to the repo are considered.
    """
    repo_root = Path(__file__).resolve().parent.parent
    for ref in (f"issue-{issue_num}", f"origin/issue-{issue_num}"):
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        if result.returncode == 0:
            return ref
    return None


def issue_token_match(name: str, issue_num: int) -> bool:
    """True when ``name`` contains ``issue_num`` as a digit-bounded token.

    Substring matching is a false-PASS vector for low-numbered issues —
    issue 56 must NOT claim ``issue_563`` (or ``issue_456``) artifacts as
    its own. The number matches only when not flanked by another digit on
    either side (``issue_56`` / ``56_panel.json`` match; ``issue_563`` /
    ``2056`` do not).
    """
    return re.search(rf"(?<!\d){issue_num}(?!\d)", name) is not None


def filter_issue_paths(paths: list[str], issue_num: int) -> list[str]:
    """Keep paths whose top-level entry under the prefix names the issue.

    Mirrors the working-tree scan (``_working_tree_issue_entries``): a
    path matches when the path component directly under the prefix
    directory contains the issue number as a digit-bounded token (never
    as a substring of a longer number — see ``issue_token_match``).
    """
    return [
        p for p in paths if len(p.split("/")) >= 2 and issue_token_match(p.split("/")[1], issue_num)
    ]


def _working_tree_issue_entries(repo_root: Path, prefix: str, issue_num: int) -> list[Path]:
    """Glob working-tree entries under ``prefix`` that name the issue.

    The raw ``*<N>*`` globs substring-match (``*56*`` also hits
    ``issue_563``), so every candidate is re-checked with
    ``issue_token_match`` on its entry name before it can count as this
    issue's artifact.
    """
    candidates = list(repo_root.glob(f"{prefix}/*issue*{issue_num}*")) + list(
        repo_root.glob(f"{prefix}/*{issue_num}*")
    )
    # dict.fromkeys dedups the two-glob union (a dir matching both patterns
    # would otherwise double-count its files in the reported file_count).
    return list(dict.fromkeys(d for d in candidates if issue_token_match(d.name, issue_num)))


def _branch_files(issue_num: int, prefix: str) -> tuple[str | None, list[str]]:
    """List issue-matching files under ``prefix`` on the issue branch.

    Eval JSONs + figures are committed on the ``issue-<N>`` worktree branch
    and only reach the main working tree at the Step 9b auto-merge, so a
    working-tree-only scan false-misses mid-pipeline (#563). Returns
    ``(ref, matching_paths)``; ``(None, [])`` when no issue branch exists.
    """
    ref = _issue_branch_ref(issue_num)
    if ref is None:
        return None, []
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", prefix],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if result.returncode != 0:
        return ref, []
    return ref, filter_issue_paths(result.stdout.splitlines(), issue_num)


def check_git_figures(issue_num: int) -> dict:
    """Check if figures for this issue are committed to git.

    Scans the working tree first, then falls back to the ``issue-<N>``
    branch refs (artifacts land there before the Step 9b auto-merge).
    """
    repo_root = Path(__file__).resolve().parent.parent
    figure_dirs = _working_tree_issue_entries(repo_root, "figures", issue_num)

    committed_files = []
    for d in figure_dirs:
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix in (".png", ".pdf", ".svg"):
                    # Check if committed
                    result = subprocess.run(
                        ["git", "ls-files", str(f.relative_to(repo_root))],
                        capture_output=True,
                        text=True,
                        cwd=repo_root,
                    )
                    if result.stdout.strip():
                        committed_files.append(str(f.relative_to(repo_root)))

    if committed_files:
        return {
            "status": "OK",
            "url": ", ".join(committed_files),
            "file_count": len(committed_files),
        }

    # Not in the main working tree — scan the issue branch before
    # reporting a miss (#563: figures committed on issue-<N> pre-merge).
    ref, branch_paths = _branch_files(issue_num, "figures/")
    branch_figs = [p for p in branch_paths if p.endswith((".png", ".pdf", ".svg"))]
    if branch_figs:
        return {
            "status": "OK",
            "url": ", ".join(branch_figs[:5]),
            "file_count": len(branch_figs),
            "detail": f"committed on branch {ref}",
        }

    if not figure_dirs:
        # Check for any figures committed recently that reference this issue
        result = subprocess.run(
            ["git", "log", "--oneline", "-5", "--", "figures/"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        return {
            "status": "WARN",
            "url": "",
            "detail": (
                f"No figure directory matching issue {issue_num} in the working tree "
                f"or on an issue-{issue_num} branch. Recent figure commits: "
                f"{result.stdout.strip() or 'none'}"
            ),
        }
    return {
        "status": "MISSING",
        "url": "",
        "detail": (
            f"Figure dirs exist ({[str(d) for d in figure_dirs]}) but no committed "
            f".png/.pdf/.svg files (working tree or issue-{issue_num} branch)"
        ),
    }


def check_pod_weights_cleaned(pod: str, output_dir: str) -> dict:
    """Check that local model weights have been cleaned from the pod."""
    if not pod:
        return {"status": "SKIP", "url": "", "detail": "No pod specified"}

    try:
        result = subprocess.run(
            [
                "ssh",
                pod,
                f"find {output_dir} -name '*.safetensors' "
                "-o -name 'model.safetensors.index.json' 2>/dev/null | head -5",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return {
                "status": "WARN",
                "url": "",
                "detail": f"SSH failed: {result.stderr.strip()}",
            }
        remaining = result.stdout.strip()
        if remaining:
            return {
                "status": "FAIL",
                "url": "",
                "detail": f"Uncleaned weights found: {remaining}",
            }
        return {"status": "OK", "url": "", "detail": "No safetensors remaining"}
    except subprocess.TimeoutExpired:
        return {"status": "WARN", "url": "", "detail": "SSH timeout (pod may be stopped)"}
    except Exception as e:
        return {"status": "ERROR", "url": "", "detail": str(e)}


def check_eval_json(issue_num: int) -> dict:
    """Check that eval result JSONs exist locally or on the issue branch.

    Scans the working tree first, then falls back to the ``issue-<N>``
    branch refs (artifacts land there before the Step 9b auto-merge).
    """
    repo_root = Path(__file__).resolve().parent.parent
    eval_dirs = _working_tree_issue_entries(repo_root, "eval_results", issue_num)

    json_files = []
    for d in eval_dirs:
        if d.is_dir():
            json_files.extend(d.glob("*.json"))
        elif d.suffix == ".json":
            json_files.append(d)

    if json_files:
        return {
            "status": "OK",
            "url": ", ".join(str(f.relative_to(repo_root)) for f in json_files[:5]),
            "file_count": len(json_files),
        }

    # Not in the main working tree — scan the issue branch before
    # reporting a miss (#563: eval JSONs committed on issue-<N> pre-merge).
    ref, branch_paths = _branch_files(issue_num, "eval_results/")
    branch_json = [p for p in branch_paths if p.endswith(".json")]
    if branch_json:
        return {
            "status": "OK",
            "url": ", ".join(branch_json[:5]),
            "file_count": len(branch_json),
            "detail": f"committed on branch {ref}",
        }
    return {
        "status": "WARN",
        "url": "",
        "detail": (
            f"No eval JSON files found matching issue {issue_num} in the working tree "
            f"or on an issue-{issue_num} branch"
        ),
    }


def run_verification(
    issue_num: int,
    experiment_type: str | None = None,
    wandb_run: str | None = None,
    wandb_artifact: str | None = None,
    hf_model_path: str | None = None,
    hf_dataset_path: str | None = None,
    pod: str | None = None,
    output_dir: str = "/workspace/explore-persona-space/outputs",
    claimed_urls_file: str | None = None,
) -> dict:
    """Run all verification checks and return structured report.

    ``experiment_type=None`` infers the type from the task's frontmatter
    ``kind`` (see ``infer_experiment_type``); an explicit value wins.

    When the caller declares no single ``wandb_run`` / ``hf_model_path``
    (the sweep case — there is no single path), the training-only rows
    fall back to the task's ``epm:results`` reproducibility card, merged
    across all epm:results markers newest-wins per field (per-cell
    ``adapter_paths`` + ``wandb_run_names`` — #608; empty resume-pass
    cards do not shadow earlier declarations — #601). Explicit
    declarations always win unchanged.
    """
    experiment_type_source = "cli"
    if experiment_type is None:
        experiment_type, experiment_type_source = infer_experiment_type(issue_num)
    report = {
        "issue": issue_num,
        "experiment_type": experiment_type,
        "experiment_type_source": experiment_type_source,
        "verdict": "PASS",
        "checks": {},
    }

    # Sweep fallback (#608): load the reproducibility_card only when a
    # training row would otherwise hard-MISS for lack of a declared path.
    results_card: dict | None = None
    if experiment_type == "training" and (not wandb_run or not hf_model_path):
        results_card = _load_results_card(issue_num)

    # 1. Eval JSON (always required)
    report["checks"]["eval_json"] = check_eval_json(issue_num)

    # 2. WandB run (always required for training)
    if wandb_run:
        report["checks"]["wandb_run"] = check_wandb_run(wandb_run)
    elif experiment_type == "training":
        card_check = check_wandb_from_card(results_card) if results_card else None
        if card_check is None:
            # Nothing declared anywhere (#608 follow-up): probe the
            # conventional <default_entity>/issue<N> project for
            # issue<N>_* runs before hard-MISSING. Fail-soft — a probe
            # error keeps the strict MISSING row below.
            card_check = check_wandb_runs_convention_project(issue_num)
        report["checks"]["wandb_run"] = card_check or {
            "status": "MISSING",
            "url": "",
            "detail": (
                "No WandB run path provided (no epm:results "
                "reproducibility_card declares wandb_run_path / "
                "wandb_run_names, and the conventional-project probe "
                f"found no issue{issue_num}_* runs)"
            ),
        }

    # 3. WandB artifact (eval results)
    if wandb_artifact:
        report["checks"]["wandb_artifact"] = check_wandb_artifact(wandb_artifact)

    # 4. HF model (training experiments)
    if experiment_type == "training":
        if hf_model_path:
            report["checks"]["hf_model"] = check_hf_hub_path(HF_MODEL_REPO, hf_model_path, "model")
        else:
            card_check = check_hf_model_from_card(results_card) if results_card else None
            report["checks"]["hf_model"] = card_check or {
                "status": "MISSING",
                "url": "",
                "detail": (
                    "No HF model path provided (required for training "
                    "experiments; no epm:results reproducibility_card "
                    "declares adapter_paths / hf_model_path either)"
                ),
            }

    # 5. HF dataset (if new data was generated)
    if hf_dataset_path:
        report["checks"]["hf_dataset"] = check_hf_hub_path(HF_DATA_REPO, hf_dataset_path, "dataset")

    # 6. Figures committed to git
    report["checks"]["figures"] = check_git_figures(issue_num)

    # 7. Pod weights cleaned (training experiments)
    if experiment_type == "training" and pod:
        report["checks"]["pod_cleanup"] = check_pod_weights_cleaned(pod, output_dir)

    # 8. Claimed-URL HEAD-check (phantom-checkpoint detection — #456).
    # Every HF/WandB URL named in the epm:results marker AND the body's
    # ## Reproducibility section MUST actually resolve at its cited revision
    # before this experiment can advance. A sentinel naming a URL string is
    # NOT evidence the underlying files exist; trusting the string is the
    # exact gap that let #456 reach awaiting_promotion with no real
    # checkpoint on HF Hub.
    report["checks"]["claimed_urls"] = check_claimed_urls_resolve(claimed_urls_file)

    # Compute overall verdict
    statuses = [c["status"] for c in report["checks"].values()]
    if (
        any(s == "FAIL" for s in statuses)
        or any(s == "MISSING" for s in statuses)
        or any(s == "ERROR" for s in statuses)
    ):
        report["verdict"] = "FAIL"
    elif any(s == "WARN" for s in statuses):
        report["verdict"] = "WARN"

    return report


def format_report(report: dict) -> str:
    """Format the verification report as markdown for a GitHub comment."""
    lines = [
        f"## Upload Verification — Issue #{report['issue']}",
        "",
        f"**Verdict: {report['verdict']}**",
        f"**Experiment type:** {report['experiment_type']}"
        f" (source: {report.get('experiment_type_source', 'cli')})",
        "",
        "| Artifact | Status | URL / Detail |",
        "|----------|--------|-------------|",
    ]

    status_emoji = {
        "OK": "PASS",
        "MISSING": "FAIL",
        "FAIL": "FAIL",
        "WARN": "WARN",
        "ERROR": "ERROR",
        "SKIP": "SKIP",
    }

    for name, check in report["checks"].items():
        display_name = name.replace("_", " ").title()
        status = status_emoji.get(check["status"], check["status"])
        detail = check.get("url") or check.get("detail", "")
        if len(detail) > 80:
            detail = detail[:77] + "..."
        lines.append(f"| {display_name} | {status} | {detail} |")

    if report["verdict"] == "FAIL":
        lines.extend(
            [
                "",
                "**Missing artifacts must be uploaded before interpretation can begin.**",
            ]
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Verify experiment artifact uploads")
    parser.add_argument("--issue", type=int, required=True, help="GitHub issue number")
    parser.add_argument(
        "--type",
        choices=["training", "eval-only", "generation", "analysis"],
        default=None,
        help=(
            "Experiment type (determines which checks are required). Omitted: "
            "inferred from the task's frontmatter `kind` (analysis/infra/batch/"
            "survey skip the training-only rows; kind=experiment conservatively "
            "assumes training, so pass --type eval-only explicitly for eval-only "
            "experiments — #563)"
        ),
    )
    parser.add_argument("--wandb-run", help="WandB run path (entity/project/runs/id)")
    parser.add_argument("--wandb-artifact", help="WandB artifact path")
    parser.add_argument("--hf-model", help="HF Hub model path within repo")
    parser.add_argument("--hf-dataset", help="HF Hub dataset path within repo")
    parser.add_argument("--pod", help="Pod name for cleanup verification")
    parser.add_argument("--output-dir", default="/workspace/explore-persona-space/outputs")
    parser.add_argument(
        "--claimed-urls-file",
        help=(
            "Path to a text file containing the epm:results marker body + "
            "the body's ## Reproducibility section. Every HF/WandB URL in "
            "the blob is HEAD-checked against its cited revision. A "
            "claimed-but-absent URL FAILs verification (phantom-checkpoint "
            "gate — see upload-verifier.md and CLAUDE.md Gotchas #456)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--no-fail", action="store_true", help="Don't exit with error on FAIL")

    args = parser.parse_args()

    report = run_verification(
        issue_num=args.issue,
        experiment_type=args.type,
        wandb_run=args.wandb_run,
        wandb_artifact=args.wandb_artifact,
        hf_model_path=args.hf_model,
        hf_dataset_path=args.hf_dataset,
        pod=args.pod,
        output_dir=args.output_dir,
        claimed_urls_file=args.claimed_urls_file,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))

    if report["verdict"] == "FAIL" and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
