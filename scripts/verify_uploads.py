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

    # Out-root residue sweep (#2187): diff a pod-side `find <out-root> -type f`
    # listing against the union of HF prefixes + ISSUE-SCOPED git trees +
    # declared discards. Any file with no permanent home FAILs — a NAME-SET
    # diff, never a count (a matching count is not a matching set; #2162).
    # A basename matched ONLY by the issue-scoped git arm is additionally
    # content-disambiguated (#2359: git blob sha1 vs the committed
    # candidates when the disk bytes are locally readable; a pod-side row
    # with no local bytes degrades to WARN `outroot-residue-basename-git-only`
    # so the exploratory pass byte-checks — a sibling leg's committed
    # same-named file must never silently cover this leg's file, #2333).
    uv run python scripts/verify_uploads.py --issue 42 \
        --outroot-listing /tmp/issue-42-outroot.txt \
        --hf-prefix issue42_slug/raw_completions

    # Realized row-count reconciliation (#2148): count what is REALLY in the
    # store's own row_index*.jsonl files and gate on the DISTINCT count of
    # the full row key vs the run's INPUT-side declaration - never a raw
    # line count and never the producer's self-reported count field (#2091:
    # `capture_rows` echoed the expectation back; PASS at a ~25% shortfall).
    uv run python scripts/verify_uploads.py --issue 42 \
        --expected-rows greedy_wildchat=2000 \
        --row-index-hf-prefix issue42_slug/store/greedy_wildchat \
        --row-index-distinct-key context_id,rollout_k

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
a declaration (see ``merged_results_card``). Nor — for the structured-
contract fields ``adapter_paths`` / ``wandb_run_names`` — does a
non-dict/non-list prose pointer when a structural declaration exists in
any marker; the prose value is kept only as a last resort so the #612
prose diagnostic still fires (#1489).

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
import fnmatch
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

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

# Out-root residue sweep (#2187): scratch classes excluded from the disk set.
# Deliberately NO size floor anywhere in the residue check — all three #2162
# losses (pilot_gate_report.json, stage2_results.json, upload_done.json) were
# under 3 KB, exactly the class a `-size +10k`-style filter silently skips.
OUTROOT_EXEMPT_DIR_PARTS = frozenset(
    {".venv", ".git", "__pycache__", ".cache", "wandb", "hf_dl", "logs"}
)
OUTROOT_EXEMPT_SUFFIXES = frozenset({".log", ".pid", ".lock", ".tmp"})

# Realized row-count reconciliation (#2148): the WITHIN-FILE sibling of the
# out-root residue check above. Budgets bound what the check may FETCH; the
# index files are KB-scale JSONL by design, so the per-file cap is generous
# while still refusing a mis-scoped invocation that would pull a data store.
ROW_INDEX_DEFAULT_GLOB = "row_index*.jsonl"
ROW_INDEX_MAX_BYTES_DEFAULT = 16_000_000  # per file
ROW_INDEX_MAX_TOTAL_BYTES_DEFAULT = 268_435_456  # aggregate (256 MiB)
ROW_INDEX_MAX_FILES_DEFAULT = 2000
# Word characters for label/path-component boundary matching: a label is a
# component-prefix match only when the next char is NOT one of these, so
# `syc_aita` never swallows `syc_aita_v2` while `arm` DOES reach `arm-repair`
# (which the ambiguity arm then refuses as a mis-scoped label vocabulary).
_ROW_INDEX_WORD_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

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

    The listing is SCOPED server-side to ``path_in_repo`` via
    ``list_repo_tree(path_in_repo=...)`` (through
    ``hub.list_repo_files_complete``, inheriting its 504 transient-retry,
    #794/#658) — a bare full-repo ``list_repo_files`` wedges >600 s on the
    ~1M-file data repo (#920, the #833 gotcha). The tree endpoint 404s
    when ``path_in_repo`` names an exact FILE rather than a directory, so
    an ``EntryNotFoundError`` falls back to one ``HfApi.file_exists`` probe.
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import EntryNotFoundError

        from explore_persona_space.orchestrate.hub import list_repo_files_complete

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        normalized = path_in_repo.rstrip("/")
        if not normalized:
            # A falsy path would silently degrade to a FULL-repo listing —
            # the exact hang this function exists to avoid. Fail loud.
            return {"status": "ERROR", "url": "", "detail": "empty path_in_repo"}
        rev_url = revision or "main"
        try:
            matching = list_repo_files_complete(
                api,
                repo_id,
                repo_type=repo_type,
                revision=revision,
                path_in_repo=normalized,
            )
        except EntryNotFoundError:
            # Not a directory at this revision; may still be an exact file
            # (tree endpoint 404s on file paths — verified on hub 0.36.2).
            # file_exists only fires AFTER the tree call proved repo+revision
            # resolve, so its False genuinely means "file missing" (its
            # swallowing of Repository/RevisionNotFoundError is unreachable
            # here).
            if api.file_exists(repo_id, normalized, repo_type=repo_type, revision=revision):
                matching = [normalized]
            else:
                matching = []
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
    Glob-bearing claimed paths (the planned-output shape, #1482) are SKIPPED
    by that checker; their count is appended to ``detail`` on OK and FAIL
    paths alike so the skip is never silent.

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
        from explore_persona_space.orchestrate.hub import (
            hf_url_path_has_glob,
            verify_artifacts_exist,
        )

        urls = extract_claimed_urls(claimed_text_path.read_text(encoding="utf-8"))
        # Qualify bare org/repo claims with their actual repo type (#599):
        # dataset claims get the datasets/ prefix the downstream checker
        # needs; claims whose repo resolves as neither type become
        # deterministic phantoms instead of aborting the scan with ERROR.
        urls, rewritten_to_original, phantoms = resolve_claimed_repo_types(urls)
        # Glob-bearing claimed URLs are the planned-output shape (#1482):
        # verify_artifacts_exist SKIPs them (a glob cannot be existence-
        # checked literally), so disclose the skipped count in ``detail`` on
        # OK and FAIL paths alike — a glob-shaped claim must never read as a
        # silently-clean OK with zero trace. Shared glob definition:
        # hub.hf_url_path_has_glob.
        n_glob = sum(1 for u in urls if hf_url_path_has_glob(u))
        glob_clause = (
            f"; {n_glob} glob-shaped claimed URL(s) not existence-checkable — skipped"
            if n_glob
            else ""
        )
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
            return {
                "status": "OK",
                "url": str(claimed_text_path),
                "detail": detail + glob_clause,
            }
        # Report missing URLs the way the task cited them (un-rewritten).
        missing_cited = phantoms + [rewritten_to_original.get(u, u) for u in missing]
        detail = "claimed-but-absent URLs (phantom): " + "; ".join(missing_cited)
        if phantoms:
            detail += (
                " [repo resolves as neither model nor dataset — phantom repo, "
                "or private without HF_TOKEN access]"
            )
        return {"status": "FAIL", "url": "", "detail": detail + glob_clause}
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


# Card fields the epm:results sentinel contract requires as per-cell
# dicts/lists (SKILL.md Step 7; the #612 ``_prose_declaration_row`` pair,
# plus ``adapter_repo_overrides`` — #1664: once the verifier reads that
# field, a truthy prose re-post winning the fold would silently reroute
# override cells back to the default repo). A prose-pointer string here
# ("unchanged from epm:results v1 ...") is truthy, so it would win the
# newest-wins fold and shadow an older marker's real declaration (#1489).
_STRUCTURED_CARD_FIELDS = ("adapter_paths", "wandb_run_names", "adapter_repo_overrides")


def _is_structural(key: str, value) -> bool:
    """False when a structured-contract field carries a non-dict/non-list."""
    return key not in _STRUCTURED_CARD_FIELDS or isinstance(value, (dict, list))


def _fold_cards(cards: list[tuple[dict, str]]) -> tuple[dict, dict[str, str], list[str]]:
    """Newest-wins per-field fold over ``cards`` (newest first).

    Returns ``(merged, fallback_fields, bypassed_prose)``: the merged card,
    the ``field -> ts`` map of fields that fell back past the newest card,
    and the ``"field @ ts"`` list of non-structural declarations bypassed in
    favor of a structural value from another marker (#1489).
    """
    merged: dict = {}
    fallback_fields: dict[str, str] = {}
    # Newest non-structural value per structured field, kept as a LAST
    # resort (#1489): used only when no marker declares a dict/list, so a
    # prose-only history still reaches the #612 _prose_declaration_row
    # diagnostic downstream instead of degrading to a generic MISSING.
    deferred_nonstructural: dict[str, tuple[object, str, int]] = {}
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
            if not _is_structural(key, value):
                deferred_nonstructural.setdefault(key, (value, ts, pos))
                continue
            merged[key] = value
            if pos > 0:
                fallback_fields[key] = ts
    bypassed_prose: list[str] = []
    for key, (value, ts, pos) in deferred_nonstructural.items():
        if key in merged:
            bypassed_prose.append(f"{key} @ {ts}")
            continue
        merged[key] = value
        if pos > 0:
            fallback_fields[key] = ts
    return merged, fallback_fields, bypassed_prose


def merged_results_card(events: list[dict]) -> dict | None:
    """Merge reproducibility cards across ALL ``epm:results`` events.

    Multi-launch runs legitimately post several ``epm:results`` markers
    (resume relaunches, drained sentinels), and a later sentinel can carry
    an empty card that would shadow the first marker's full declaration
    (#601: a resume pass with every cell ``resumed_skip`` posted
    ``adapter_paths: {}``, masking 16 verified adapter paths). Each FIELD
    therefore resolves newest-wins: the value comes from the newest card
    that declares it non-empty (``_is_declared``). For the structured-
    contract fields (``_STRUCTURED_CARD_FIELDS``) a non-dict/non-list
    declaration additionally does not count as a declaration when a
    structural one exists in ANY card — it is kept only as a last resort
    so the #612 prose diagnostic still fires downstream (#1489: a re-post
    prose pointer shadowed an older marker's real 64-path list). When any
    field falls back past the newest card, the merged card carries a
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
    merged, fallback_fields, bypassed_prose = _fold_cards(cards)
    notes: list[str] = []
    if fallback_fields:
        notes.append(
            "field(s) declared by an earlier epm:results marker, not the latest: "
            + ", ".join(f"{k} @ {ts}" for k, ts in sorted(fallback_fields.items()))
        )
    if bypassed_prose:
        notes.append(
            "non-structural (prose) declaration(s) bypassed in favor of a "
            "structural value from another marker (#1489): " + ", ".join(sorted(bypassed_prose))
        )
    if notes:
        note = "; ".join(notes)
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


def _strip_repo_prefix(p: str, cell_repo: str) -> str:
    """Drop a leading ``<repo>/`` from a declared path (#610), where
    ``<repo>`` is the repo this path is CHECKED AGAINST, so it
    existence-checks as the in-repo subfolder it names. Paths that don't
    carry that prefix (plain in-repo paths, https URLs, other repos'
    prefixes) pass through verbatim and MISS visibly (fail-loud)."""
    prefix = cell_repo.rstrip("/") + "/"
    if p.startswith(prefix) and len(p) > len(prefix):
        return p[len(prefix) :]
    return p


def _card_repo_overrides(card: dict) -> tuple[dict[str, str], list[str]]:
    """Parse ``adapter_repo_overrides`` into ``(overrides, notes)`` (#1664).

    Valid form: a per-cell ``{cell_id: repo_id}`` dict of non-empty
    strings (#1586's overflow split). Malformed input — a non-dict value,
    non-string repo ids — never crashes and never silently drops: the
    affected cells fall through to the default repo and a visible note
    is returned for the row detail.
    """
    overrides_raw = card.get("adapter_repo_overrides")
    overrides: dict[str, str] = {}
    notes: list[str] = []  # visible diagnostics appended to the row detail
    if isinstance(overrides_raw, dict):
        overrides = {
            str(k): str(v).strip()
            for k, v in overrides_raw.items()
            if isinstance(v, str) and v.strip()
        }
        n_bad = len(overrides_raw) - len(overrides)
        if n_bad:
            notes.append(
                f"{n_bad} adapter_repo_overrides value(s) not a non-empty "
                "string — ignored (cell checked against the default repo)"
            )
    elif _is_declared(overrides_raw):
        notes.append(
            "adapter_repo_overrides is not a per-cell dict "
            f"({type(overrides_raw).__name__}) — ignored; every path "
            "checked against the default repo"
        )
    return overrides, notes


def _card_model_pairs(
    card: dict, repo: str, overrides: dict[str, str], notes: list[str]
) -> tuple[list[tuple[str, str]], dict | None]:
    """Collect the unique ``(repo, path)`` pairs a card declares (#1664).

    Dict-form ``adapter_paths`` cells resolve against
    ``overrides.get(cell, repo)``; list-form paths and ``hf_model_path``
    carry no cell key and always use the default ``repo``. Appends
    orphan-override-key / list-form-mismatch diagnostics to ``notes`` in
    place. Dedup key is the ``(repo, path)`` tuple — the same path string
    under two repos is two distinct existence checks (collapsing on the
    path alone would skip one repo's). Returns ``(pairs,
    prose_violation)``, the latter the #612 prose row when
    ``adapter_paths`` is a non-empty string.
    """
    pairs: list[tuple[str, str]] = []
    adapter_paths = card.get("adapter_paths")
    prose_violation: dict | None = None
    if isinstance(adapter_paths, dict):
        for cell, p in adapter_paths.items():
            cell_repo = overrides.get(str(cell), repo)
            pairs.append((cell_repo, _strip_repo_prefix(str(p), cell_repo)))
        orphans = sorted(set(overrides) - {str(c) for c in adapter_paths})
        if orphans:
            notes.append(
                f"{len(orphans)} adapter_repo_overrides key(s) with no "
                "matching adapter_paths cell (nothing checked for them): " + ", ".join(orphans[:5])
            )
    elif isinstance(adapter_paths, list):
        pairs.extend((repo, _strip_repo_prefix(str(p), repo)) for p in adapter_paths)
        if overrides:
            notes.append(
                "adapter_repo_overrides declared but adapter_paths is a "
                "LIST (no cell keys) — overrides unmatchable; every path "
                "checked against the default repo"
            )
    elif isinstance(adapter_paths, str) and _is_declared(adapter_paths):
        prose_violation = _prose_declaration_row("adapter_paths", adapter_paths)
    single = card.get("hf_model_path")
    if single:  # no cell key → always the default repo (see the caller)
        pairs.append((repo, _strip_repo_prefix(str(single), repo)))
    return list(dict.fromkeys(pairs)), prose_violation


def check_hf_model_from_card(card: dict) -> dict | None:
    """Verify model paths declared in an epm:results reproducibility_card.

    Accepts a per-cell ``adapter_paths`` dict/list and/or a single
    ``hf_model_path``. Every path resolves against ``hf_model_repo``
    (default ``HF_MODEL_REPO``) UNLESS the card carries a per-cell
    ``adapter_repo_overrides`` dict (``{cell_id: repo_id}`` keyed on
    ``adapter_paths`` cells — the #1108 overflow-repo split; incident
    #1586: two cells uploaded to the main repo while the card default
    was the overflow repo read as a false hf_model MISSING, #1664):
    dict-form ``adapter_paths`` cells then check against
    ``overrides.get(cell, default)``, while list-form paths and
    ``hf_model_path`` carry no cell key and always use the default
    repo. Each unique ``(repo, path)`` pair is existence-checked via
    ``check_hf_hub_path``. Declared paths prefixed with the repo id
    they are CHECKED AGAINST (#610's ``<repo>/adapters/...`` shape)
    have that leading ``<repo>/`` stripped first — passed verbatim as
    ``path_in_repo`` they can never match the repo's ``adapters/...``
    file list, which false-MISSed a fully-uploaded sweep; a path
    carrying a DIFFERENT repo's prefix passes through verbatim and
    MISSes visibly. Malformed overrides (a non-dict value, non-string
    repo ids, keys with no matching cell) never crash and never
    silently drop: affected cells check against the default repo and a
    ``NOTE:`` is appended to the detail. Merge-layer caveat: the field
    folds per-field newest-wins like ``adapter_paths`` (#601), so an
    older STRUCTURAL overrides dict persists even when a corrected
    re-post OMITS the field — re-declare it corrected instead. A
    STRING-valued ``adapter_paths`` (the #612 prose-template shape) is
    unverifiable; instead of silently ignoring it (which read as a
    generic declaration-gap MISSING on a fully-uploaded sweep), the
    row names the producer-contract violation
    (``_prose_declaration_row``). Returns ``None`` when the card
    declares no model paths (caller falls through to the MISSING row).
    """
    repo = str(card.get("hf_model_repo") or HF_MODEL_REPO)
    overrides, notes = _card_repo_overrides(card)
    pairs, prose_violation = _card_model_pairs(card, repo, overrides, notes)
    if not pairs:
        if prose_violation is not None:
            if notes:
                prose_violation["detail"] = (
                    f"{prose_violation['detail']}; NOTE: " + "; NOTE: ".join(notes)
                )
            return _append_card_provenance(prose_violation, card)
        return None

    absent: list[str] = []
    errored = False
    total_files = 0
    for cell_repo, p in pairs:
        res = check_hf_hub_path(cell_repo, p, "model")
        if res["status"] == "OK":
            total_files += res.get("file_count", 0)
        else:
            errored = errored or res["status"] == "ERROR"
            # Repo-qualify only override-repo entries — default-repo entries
            # keep the pre-#1664 detail strings byte-identical.
            label = p if cell_repo == repo else f"{cell_repo}/{p}"
            absent.append(f"{label} ({res.get('detail') or res['status']})")
    override_repos = sorted({r for r, _ in pairs if r != repo})
    where = (
        repo
        if not override_repos
        else f"{repo} (default) + override repo(s): {', '.join(override_repos)}"
    )
    if absent:
        result = {
            "status": "ERROR" if errored else "MISSING",
            "url": "",
            "detail": (
                f"reproducibility_card declares {len(pairs)} model path(s) under "
                f"{where}; unresolved: " + "; ".join(absent[:5])
            ),
            "source": "epm:results reproducibility_card",
        }
    else:
        result = {
            "status": "OK",
            "url": f"https://huggingface.co/{repo}/tree/main",
            "file_count": total_files,
            "detail": (
                f"all {len(pairs)} model path(s) from the epm:results "
                f"reproducibility_card resolve on {where}"
            ),
            "source": "epm:results reproducibility_card",
        }
    if prose_violation is not None:
        # A real hf_model_path resolved (or failed) above, but the card ALSO
        # carried an unverifiable prose adapter_paths — keep that visible.
        result["detail"] = f"{result['detail']}; ALSO: {prose_violation['detail']}"
    if notes:
        result["detail"] = f"{result['detail']}; NOTE: " + "; NOTE: ".join(notes)
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


def _unconsulted_prose_wandb_note(card: dict | None) -> str | None:
    """Detail note when a card's bare ``wandb`` field is consulted by NO check (#612).

    ``check_wandb_from_card`` reads ``wandb_run_path`` / ``wandb_run`` /
    ``wandb_run_names`` only; a free-text line stored under the bare
    ``wandb`` key (the #612 prose-template card) is silently skipped, so
    the producer-contract violation stayed invisible unless the
    adapter_paths side also violated. Returns text for ``run_verification``
    to APPEND to whichever fallback row wins the wandb_run slot —
    diagnostic-only, never a row of its own, never a status change.
    """
    value = (card or {}).get("wandb")
    if not isinstance(value, str) or not _is_declared(value):
        return None
    snippet = value if len(value) <= 100 else value[:97] + "..."
    return (
        "card carries an unconsulted prose 'wandb' field — producers must "
        "declare wandb_run_path / wandb_run_names (epm:results sentinel "
        f"contract, .claude/skills/issue/SKILL.md Step 7; incident #612): {snippet!r}"
    )


def _verifier_repo_root() -> Path:
    """Repo root the git arms run in (the checkout containing this script).

    Module-level seam (#2359): the hermetic residue tests monkeypatch this to
    a temp git repo so the REAL subprocess git arm runs against real committed
    fixtures — the mid-flight live tree is not a stable fixture.
    """
    return Path(__file__).resolve().parent.parent


def _issue_branch_ref(issue_num: int) -> str | None:
    """Return the first existing git ref for the issue branch, or None.

    Prefers the local worktree branch (``issue-<N>``) over the pushed
    remote-tracking ref (``origin/issue-<N>``). No fetch is performed —
    only refs already known to the repo are considered.
    """
    repo_root = _verifier_repo_root()
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


def _issue_path_token_re(issue_num: int) -> re.Pattern[str]:
    """Path-component issue-token filter for the residue check's git arm (#2187).

    ``.search`` semantics over a ``(?:^|/)`` anchor: a path matches when some
    path COMPONENT begins with the issue token at a non-digit boundary.
    ``eval_results/issue_<N>/...``, ``figures/issue_<N>/...``,
    ``scripts/issue<N>_*.py``, ``docs/methodology/issue_<N>.md`` all match;
    ``eval_results/issue_1739/...`` for issue 2162 does not (different issue);
    ``issue_21620`` does not (digit boundary); ``myissue_2162`` does not (the
    component must BEGIN with the token).
    """
    return re.compile(rf"(?:^|/)issue[-_]?{issue_num}(?![0-9])")


def filter_issue_scoped_git_paths(paths: list[str], issue_num: int) -> list[str]:
    """Keep tree paths carrying the issue token as a path component (#2187).

    NEVER feed the unfiltered tree to a basename join: conventional filenames
    collide across issues (measured at HEAD, 2026-08-08:
    ``pilot_gate_report.json`` at 8 cross-issue paths and ``upload_done.json``
    at 4), so a whole-tree basename arm false-PASSes exactly the losses the
    out-root residue check exists to catch.
    """
    pattern = _issue_path_token_re(issue_num)
    return [p for p in paths if pattern.search(p)]


class _GitCandidate(NamedTuple):
    """One committed same-basename candidate on the issue-scoped git arm."""

    path: str
    oid: str
    size: int


def _git_tree_candidates_for_issue(issue_num: int) -> dict[str, list[_GitCandidate]]:
    """Basename -> committed candidates (path, blob OID, size) for the issue.

    Supersedes the former basename-SET helper (#2359): the residue check's
    git arm needs each candidate's blob OID + byte size so a basename match
    can be CONTENT-disambiguated — a sibling leg's committed same-named file
    must not cover this leg's different-bytes file (#2333 cross-leg
    false-OK). Reads the refs (never the working tree — sparse-worktree
    safe) via ``git ls-tree -r -l`` (long format: ``<mode> <type> <oid>
    <size>\\t<path>``), keeps blob rows whose path passes
    :func:`filter_issue_scoped_git_paths`, and dedups on (path, oid) across
    the issue branch + HEAD (a candidate differing BETWEEN refs contributes
    both versions — either committed content is a permanent home).
    Uncommitted working-tree files are deliberately NOT counted (uncommitted
    is not permanent — matches Step 2.9's posture). Raises ``RuntimeError``
    on a failed listing or a STRUCTURALLY malformed row — one that cannot be
    split into ``<mode> <type> <oid> <size>\\t<path>`` (fail-loud; the caller
    surfaces it as an ERROR row, which flips the overall verdict to FAIL).
    Successfully-parsed non-blob rows (tree/commit entries) are skipped as
    legitimate non-candidates.
    """
    repo_root = _verifier_repo_root()
    refs: list[str] = []
    branch_ref = _issue_branch_ref(issue_num)
    if branch_ref is not None:
        refs.append(branch_ref)
    refs.append("HEAD")
    candidates: dict[str, list[_GitCandidate]] = {}
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "-l", ref],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git ls-tree {ref} failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        parsed: dict[str, tuple[str, int]] = {}
        for line in result.stdout.splitlines():
            # A STRUCTURAL parse failure (no tab separator / wrong metadata
            # field count) is a row the verdict must not silently build on
            # — fail loud naming the ref + the offending row (#2359 r2).
            meta, sep, path = line.partition("\t")
            if not sep:
                raise RuntimeError(
                    f"unparseable git ls-tree -l row (no tab separator) for {ref}: {line!r}"
                )
            fields = meta.split()
            if len(fields) != 4:
                raise RuntimeError(
                    f"unparseable git ls-tree -l row (expected 4 metadata fields, "
                    f"got {len(fields)}) for {ref}: {line!r}"
                )
            _mode, otype, oid, size_field = fields
            if otype != "blob":
                continue  # parsed non-blob row (tree/commit entry) — no content to compare
            try:
                size = int(size_field)
            except ValueError as e:
                # A blob row always carries a numeric size; anything else is
                # a parse the verdict must not silently build on.
                raise RuntimeError(f"unparseable git ls-tree -l blob row: {line!r}") from e
            parsed[path] = (oid, size)
        for path in filter_issue_scoped_git_paths(list(parsed), issue_num):
            oid, size = parsed[path]
            if (path, oid) in seen:
                continue
            seen.add((path, oid))
            basename = path.rsplit("/", 1)[-1]
            candidates.setdefault(basename, []).append(_GitCandidate(path, oid, size))
    return candidates


def _git_blob_sha1(path: str) -> str:
    """Git blob SHA-1 of the file at ``path`` (matches ``git hash-object``).

    hashlib sha1 over the blob header ``b"blob %d\\0" % nbytes`` + the raw
    file bytes, chunked so a large file never materializes in memory. Valid
    for SHA-1 object-format repos (this repo is one), and the committed blob
    OIDs the residue check compares against are unaffected by LFS/text
    filters (none configured in ``.gitattributes`` for these paths). Raises
    ``OSError`` on an unreadable path — the caller maps that to residue
    (fail-toward-FAIL).
    """
    h = hashlib.sha1(b"blob %d\0" % os.stat(path).st_size)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hf_prefix_basenames(hf_prefixes: tuple[str, ...]) -> set[str]:
    """Basenames of files under each caller-supplied HF data-repo prefix.

    Prefix-scoped listings only (a bare full-repo ``list_repo_files`` wedges
    over 600 s on the ~1M-file data repo — #920); prefixes are issue-scoped by
    construction (``issue<N>_<slug>/...``), so no token filter is needed on
    this arm. A prefix that does not resolve contributes NOTHING — its files
    then read as residue, the safe (fail-toward-FAIL) direction; any OTHER
    listing failure propagates (fail-loud -> ERROR row at the caller).
    """
    from huggingface_hub import HfApi
    from huggingface_hub.utils import EntryNotFoundError

    from explore_persona_space.orchestrate.hub import list_repo_files_complete

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    basenames: set[str] = set()
    for prefix in hf_prefixes:
        normalized = str(prefix).rstrip("/")
        if not normalized:
            continue
        try:
            files = list_repo_files_complete(
                api,
                HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=normalized,
            )
        except EntryNotFoundError:
            # Prefix absent on the repo: zero permanent homes under it. The
            # unmatched disk files then surface as residue (FAIL) rather than
            # ERROR — the remediation is uploading/declaring, not debugging.
            continue
        basenames.update(f.rsplit("/", 1)[-1] for f in files)
    return basenames


def _file_size_or_unknown(path: str) -> str:
    """Byte size of ``path`` when it resolves locally, else a marker string.

    A pod-side ``--outroot-listing`` capture names pod paths this VM cannot
    stat; the residue detail still names every file, just without the size.
    """
    try:
        return f"{os.stat(path).st_size} B"
    except OSError:
        return "size unknown - pod-side listing"


def _outroot_disk_entries(
    outroot_listing: str | None, outroot: str | None
) -> list[tuple[str, str]] | dict:
    """Build the disk-entry list ``(full path, root-relative path)``, or an ERROR row.

    Missing-input handling is fail-loud: a nonexistent listing file or
    out-root directory returns an ERROR row dict (never a silent empty list —
    ``disk=0`` on missing input would false-PASS the residue gate). The
    root-relative second element is what exemption matching keys on: listing
    mode strips the listing's common directory prefix so both input modes
    match the same root-relative shape.
    """
    entries: list[tuple[str, str]] = []
    if outroot_listing:
        listing_path = Path(outroot_listing)
        if not listing_path.is_file():
            return {
                "status": "ERROR",
                "url": "",
                "detail": (
                    f"out-root listing file not found: {outroot_listing} - a missing "
                    "input never reads as a clean sweep (fail-loud); re-capture via "
                    "find <out-root> -type f | sort"
                ),
            }
        fulls = [
            line.strip()
            for line in listing_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if fulls:
            try:
                common = os.path.commonpath([os.path.dirname(f) or "." for f in fulls])
            except ValueError as e:
                return {
                    "status": "ERROR",
                    "url": "",
                    "detail": (
                        "out-root listing is malformed (mixed absolute/relative "
                        f"paths): {type(e).__name__}: {e}"
                    ),
                }
            for full in fulls:
                entries.append((full, os.path.relpath(full, common)))
    else:
        root = Path(outroot)  # type: ignore[arg-type]
        if not root.is_dir():
            return {
                "status": "ERROR",
                "url": "",
                "detail": (
                    f"out-root directory not found: {outroot} - a missing path never "
                    "reads as disk=0 OK (fail-loud); an out-root that exists and is "
                    "empty is the legitimate disk=0 case"
                ),
            }
        for p in sorted(root.rglob("*")):
            if p.is_file():
                entries.append((str(p), str(p.relative_to(root))))
    return entries


def _match_outroot_files(
    disk_paths: list[str],
    covered_names: set[str],
    git_candidates: dict[str, list[_GitCandidate]],
) -> tuple[list[str], list[str], int]:
    """Per-file matching for :func:`check_outroot_residue` (#2359).

    HF/discard coverage first (``covered_names``), then the issue-scoped git
    arm with content disambiguation. Returns ``(residue descriptions,
    git-only-unverified descriptions, content-verified count)``.
    """
    residue: list[str] = []
    git_only_unverified: list[str] = []
    content_verified = 0
    for full in disk_paths:
        basename = full.rsplit("/", 1)[-1]
        if basename in covered_names:
            continue
        candidates = git_candidates.get(basename)
        if candidates is None:
            residue.append(f"{full} ({_file_size_or_unknown(full)})")
            continue
        cand_paths = ", ".join(c.path for c in candidates)
        if not Path(full).is_file():
            # No local bytes (pod-side --outroot-listing row): the git-only
            # basename match is UNVERIFIABLE here — collect for the WARN.
            git_only_unverified.append(f"{full} ~ {cand_paths}")
            continue
        # Local access: disambiguate by content. Size equality is the cheap
        # first pass (no hashing when no candidate matches the byte size);
        # an unreadable file is residue (fail-toward-FAIL).
        try:
            disk_size = os.stat(full).st_size
            same_size = [c for c in candidates if c.size == disk_size]
            disk_oid = _git_blob_sha1(full) if same_size else None
        except OSError as e:
            residue.append(
                f"{full} (content check failed: {type(e).__name__}: {e}; "
                f"committed candidate(s): {cand_paths})"
            )
            continue
        if same_size and any(c.oid == disk_oid for c in same_size):
            content_verified += 1
            continue
        residue.append(
            f"{full} ({disk_size} B) same basename, different content - "
            f"committed candidate(s): {cand_paths}"
        )
    return residue, git_only_unverified, content_verified


def check_outroot_residue(
    issue_num: int,
    *,
    outroot_listing: str | None = None,
    outroot: str | None = None,
    hf_prefixes: tuple[str, ...] = (),
    exempt_globs: tuple[str, ...] = (),
    discarded_names: tuple[str, ...] = (),
) -> dict:
    """Name-set diff of the run's out-root files vs their permanent homes (#2187).

    ``residue = names(disk) - (names(HF prefixes UNION issue-scoped git trees)
    UNION declared_discards UNION exemptions)``. Matching key is the BASENAME
    (a pod path ``/workspace/issue2162_out/upload_done.json`` and an HF path
    ``issue2162_slug/margin/upload_done.json`` share no path structure), and
    the git arm is ISSUE-SCOPED — see :func:`filter_issue_scoped_git_paths`
    for why the whole tree is never consulted. Counts are context only: a
    matching count is not a matching set (#2162: 236 pod files vs 235
    uploaded read clean on counts while a file was lost).

    Per-file matching, HF/discards FIRST (#2359 — the order is load-bearing):
    a basename resolving at an HF prefix or a declared discard is covered
    with no further check; a basename resolving ONLY via the issue-scoped
    git arm is CONTENT-disambiguated, because the git arm is issue-scoped
    but not leg-scoped and a sibling leg's committed same-named file must
    not cover this leg's unpersisted file (#2333: leg-B's
    ``upload_done.json`` read OK off leg-A's committed different-bytes
    copy). With local access the disk file's git blob sha1 is compared
    against the committed candidates' OIDs (size equality as the cheap
    first pass — no hashing when no candidate matches the byte size): an
    OID match is covered; no match is residue ("same basename, different
    content", naming both paths); an unreadable file is residue
    (fail-toward-FAIL). Without local access (a pod-side
    ``--outroot-listing`` row) the git-only match is UNVERIFIABLE here —
    the check degrades to WARN carrying the literal token
    ``outroot-residue-basename-git-only`` plus both paths, so the
    upload-verifier's exploratory pass knows to byte-check. Per-file LOCAL
    ACCESS (not the input-mode flag) selects content-check vs WARN, so a
    VM-side listing still gets the strong check.

    Verdicts: residue non-empty -> FAIL naming every residue path + size
    (git-only-unverified files, when also present, are named in the same
    detail — residue dominates); residue empty but git-only-unverified
    non-empty -> WARN (token above); both empty -> OK (detail carries the
    ``content-verified=<n>`` git-arm-verified count); neither
    ``outroot_listing`` nor ``outroot`` supplied -> SKIP (legacy invocations
    unchanged); a git/HF listing failure -> ERROR (fail-loud; ERROR flips
    the overall verdict to FAIL). An empty ``hf_prefixes`` with a listing
    supplied still runs, with a WARN-worded detail (fail-toward-FAIL:
    HF-resident files then read as residue).

    Missing-input handling (fail-loud): a nonexistent ``outroot_listing``
    file or ``outroot`` directory returns ERROR — a missing input must never
    read as a clean ``disk=0`` OK (the silent-default false-PASS is the exact
    failure class this check exists to close). An ``outroot`` directory that
    EXISTS and is genuinely empty is the legitimate ``disk=0`` OK.

    Exemption parity across input modes: exemption dir-parts and caller globs
    match the ROOT-RELATIVE path in BOTH modes — listing mode strips the
    listing's common directory prefix first, so an out-root that is itself
    nested under an exempt-named directory (``/workspace/logs/<out>/...``)
    is never wholesale-exempted (that would be a false-PASS on this gate).
    """
    if not outroot_listing and not outroot:
        return {
            "status": "SKIP",
            "url": "",
            "detail": (
                "no out-root listing supplied (--outroot-listing/--outroot); "
                "out-root residue not checked"
            ),
        }

    # 1. Disk set. NO size floor (#2162: all three losses were <3 KB).
    # Exemptions key on the ROOT-RELATIVE path in BOTH modes: listing mode
    # strips the common directory prefix so a full pod path's own components
    # (/workspace/logs/<out>/...) can never wholesale-exempt the listing.
    # Missing input -> ERROR row (never a silent disk=0 OK).
    entries = _outroot_disk_entries(outroot_listing, outroot)
    if isinstance(entries, dict):
        return entries

    disk_paths: list[str] = []
    for full, rel in entries:
        parts = rel.split("/")
        if any(part in OUTROOT_EXEMPT_DIR_PARTS for part in parts[:-1]):
            continue
        if any(rel.endswith(suffix) for suffix in OUTROOT_EXEMPT_SUFFIXES):
            continue
        if any(
            fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(parts[-1], pat) for pat in exempt_globs
        ):
            continue
        disk_paths.append(full)

    warn_prefix = ""
    if not hf_prefixes:
        warn_prefix = (
            "WARNING: no --hf-prefix supplied - HF permanent homes were NOT "
            "consulted (fail-toward-FAIL: HF-resident files read as residue); "
        )

    # 2. Permanent-home sets. ORDER IS LOAD-BEARING (#2359): HF-arm basenames
    # + declared discards cover a file BEFORE any git-arm content logic runs,
    # so a both-arms match in listing mode stays a clean OK (no spurious WARN
    # on healthy runs whose files are both uploaded and committed). The git
    # arm carries per-candidate blob OID + size for the content check below.
    # A listing failure on either arm is surfaced as ERROR — never swallowed
    # into a false OK.
    try:
        covered_names = _hf_prefix_basenames(tuple(hf_prefixes))
        git_candidates = _git_tree_candidates_for_issue(issue_num)
    except Exception as e:
        return {
            "status": "ERROR",
            "url": "",
            "detail": (f"{warn_prefix}permanent-home listing failed: {type(e).__name__}: {e}"),
        }
    covered_names.update(discarded_names)

    # 3. Verdict: the per-file name-set diff (+ git-arm content check), never
    # the counts. residue -> FAIL; git-only matches with no local bytes ->
    # WARN (`outroot-residue-basename-git-only`); residue dominates the WARN.
    residue, git_only_unverified, content_verified = _match_outroot_files(
        disk_paths, covered_names, git_candidates
    )
    matched = len(disk_paths) - len(residue) - len(git_only_unverified)
    if residue:
        detail = (
            f"{warn_prefix}{len(residue)} file(s) match no permanent home "
            f"(HF prefixes + issue-scoped git trees + declared discards): "
            f"{', '.join(residue)}. A matching count is not a matching set - "
            f"the verdict is the name-set diff, never the counts."
        )
        if git_only_unverified:
            detail += (
                f" Additionally {len(git_only_unverified)} git-only basename "
                f"match(es) with no local bytes to compare (byte-check in the "
                f"exploratory pass): {'; '.join(git_only_unverified)}"
            )
        return {"status": "FAIL", "url": "", "detail": detail}
    if git_only_unverified:
        return {
            "status": "WARN",
            "url": "",
            "detail": (
                f"{warn_prefix}outroot-residue-basename-git-only: "
                f"{len(git_only_unverified)} file(s) matched ONLY issue-scoped "
                f"git basenames and have no local bytes to compare - "
                f"byte-check in the exploratory pass: "
                f"{'; '.join(git_only_unverified)}"
            ),
        }
    return {
        "status": "OK",
        "url": "",
        "detail": (
            f"{warn_prefix}disk={len(disk_paths)} matched={matched}; "
            f"content-verified={content_verified}; "
            f"verdict is the name-set diff, never the counts"
        ),
    }


def _label_matches_component(label: str, component: str) -> bool:
    """Component-boundary label match (#2148 attribution step).

    True when the path component IS the label, or starts with it followed by
    a NON-word character: `syc_aita` must never swallow `syc_aita_v2` (the
    boundary char `_` is a word char), while `arm` DOES match `arm-repair`
    (`-` is not) - the ambiguity ERROR arm exists to refuse exactly that
    nested-label vocabulary rather than guess.
    """
    if component == label:
        return True
    if label and component.startswith(label):
        rest = component[len(label) :]
        return bool(rest) and rest[0] not in _ROW_INDEX_WORD_CHARS
    return False


def _labels_for_path(labels: tuple[str, ...], rel_path: str) -> list[str]:
    """All declared labels matching any path component of ``rel_path``."""
    comps = [c for c in rel_path.split("/") if c]
    return [lb for lb in labels if any(_label_matches_component(lb, c) for c in comps)]


def _row_index_resolve_revision() -> str:
    """Resolve ONE data-repo revision at check entry (#2148 round 2).

    The ``stage_hub_prefix`` pattern (hub.py: retried ``repo_info(...).sha``):
    the listing walks, the batched size probe, and every staged fetch read
    that SAME SHA, so a verdict is traceable to one Hub snapshot - a commit
    landing between listing and fetch can neither fuse files from different
    commits into one verdict nor grow past the byte caps checked off the
    listing. Seam for tests.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = retry_transient(
        lambda: api.repo_info(HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info({HF_DATA_REPO})",
    )
    return str(info.sha)


def _row_index_hf_entries(
    hf_prefixes: tuple[str, ...], *, revision: str | None
) -> list[tuple[str, int | None]]:
    """One scoped tree walk per DISTINCT prefix, pinned to ``revision``.

    Seam for tests (monkeypatched); the live body reuses the canonical
    sizes-preserving walker (`list_repo_entries_complete`, #833 pagination) -
    never a bare full-repo `list_repo_files` (wedges on the ~1M-file repo,
    #920). Prefixes are canonicalized (trailing-slash strip) and exact
    duplicates walk ONCE; OVERLAPPING (parent/child) prefixes still re-list
    shared paths - the caller dedupes entries by (mode, path) (#2148 round
    2). A prefix that does not resolve raises (fail-loud -> ERROR row at
    the caller): unlike the residue check there is no fail-toward-FAIL
    direction here - a silently-empty prefix would read as `row-index-missing`
    with a store-defect remediation when the actual defect is the invocation.
    A prefix that canonicalizes to EMPTY (``""`` / ``"/"``) raises for the
    same reason (#2148 round 3): silently skipping it would blame the store
    (`row-index-missing`) for an operator's malformed flag. The check
    validates this at entry too (`row-index-prefix-empty`, zero Hub reads);
    this raise is defense in depth for direct callers.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_repo_entries_complete

    empty = [repr(p) for p in hf_prefixes if not str(p).rstrip("/")]
    if empty:
        raise ValueError(
            f"--row-index-hf-prefix canonicalizes to an empty prefix: {', '.join(empty)} "
            "- a malformed flag is an invocation defect, never a store defect"
        )
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    entries: list[tuple[str, int | None]] = []
    seen_prefixes: set[str] = set()
    for prefix in hf_prefixes:
        normalized = str(prefix).rstrip("/")
        if normalized in seen_prefixes:
            continue
        seen_prefixes.add(normalized)
        entries.extend(
            list_repo_entries_complete(
                api,
                HF_DATA_REPO,
                repo_type="dataset",
                revision=revision,
                path_in_repo=normalized,
            )
        )
    return entries


def _row_index_resolve_sizes(paths: list[str], *, revision: str | None) -> dict[str, int | None]:
    """ONE batched ``get_paths_info`` POST for listing entries whose size the
    tree walk left unknown (#2148 budget step), pinned to ``revision`` and
    riding ``retry_transient`` - a verify-path Hub call is retried, never a
    one-transient-429-fails-a-healthy-store probe (upload-policy #1335 r5;
    #2148 round 2). Seam for tests."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    infos = retry_transient(
        lambda: api.get_paths_info(
            HF_DATA_REPO, paths, expand=True, repo_type="dataset", revision=revision
        ),
        what=f"get_paths_info(realized row counts, {len(paths)} path(s))",
    )
    return {getattr(i, "path", ""): getattr(i, "size", None) for i in infos}


def _row_index_fetch(path_in_repo: str, target: Path, *, revision: str | None) -> Path:
    """Fetch ONE budget-passed index file at the pinned ``revision`` via the
    retried atomic staging helper (`stage_hub_file` - transport-retried,
    fail-loud). Seam for tests."""
    from explore_persona_space.orchestrate.hub import stage_hub_file

    return stage_hub_file(
        HF_DATA_REPO, path_in_repo, target, repo_type="dataset", revision=revision
    )


def _row_index_entries(
    hf_prefixes: tuple[str, ...],
    local_root: str | None,
    glob_pattern: str,
    *,
    revision: str | None,
) -> list[dict] | dict:
    """Enumerate candidate row-index files, or return an ERROR row dict.

    Entry shape: ``{"path": <fetch id>, "rel": <attribution path>, "size":
    int | None, "mode": "hf" | "local"}``. Missing-input handling is
    fail-loud (a nonexistent local root / failed Hub listing is never a
    clean zero - the silent-default false-PASS is the #2091 class).

    The returned set is DEDUPLICATED by ``(mode, path)`` (#2148 round 2):
    repeated / overlapping ``--row-index-hf-prefix`` values re-list shared
    paths, and a doubled entry would double every line count (flipping a
    keyless `realized-rows-short` FAIL into the nonblocking
    `realized-rows-no-distinct-key` WARN) and corrupt the file/byte budget
    sums. (The caller additionally REFUSES mixed local+HF sources at check
    entry - #2148 round 3 - so in practice every entry shares one mode.)
    Agreeing duplicates count ONCE, and an UNKNOWN size coalesces with a
    known one (the tree walk legitimately returns ``size=None`` -
    ``list_repo_entries_complete`` is ``int | None`` - and this check's own
    budget path treats None as "unknown, probe it", so a (None, int) pair is
    not a conflict: a known size fills a prior None, a later None never
    displaces a known value). Only a same-path duplicate with CONFLICTING
    KNOWN sizes is a real listing inconsistency and ERRORs
    (`row-index-duplicate-conflict`) - never silently deduplicated.
    """
    entries: list[dict] = []
    if local_root:
        root = Path(local_root)
        if not root.is_dir():
            return {
                "status": "ERROR",
                "url": "",
                "detail": (
                    f"row-index local root not found: {local_root} - a missing "
                    "input never reads as a clean zero (fail-loud)"
                ),
            }
        for p in sorted(root.rglob("*")):
            if not p.is_file() or not fnmatch.fnmatch(p.name, glob_pattern):
                continue
            try:
                size = p.stat().st_size
            except OSError as e:
                return {
                    "status": "ERROR",
                    "url": "",
                    "detail": f"row-index file unreadable: {p} ({type(e).__name__}: {e})",
                }
            entries.append(
                {"path": str(p), "rel": str(p.relative_to(root)), "size": size, "mode": "local"}
            )
    if hf_prefixes:
        try:
            listed = _row_index_hf_entries(tuple(hf_prefixes), revision=revision)
        except Exception as e:
            return {
                "status": "ERROR",
                "url": "",
                "detail": f"row-index Hub listing failed: {type(e).__name__}: {e}",
            }
        for path, size in listed:
            if fnmatch.fnmatch(path.rsplit("/", 1)[-1], glob_pattern):
                entries.append({"path": path, "rel": path, "size": size, "mode": "hf"})

    deduped: list[dict] = []
    first_seen: dict[tuple[str, str], dict] = {}
    for entry in entries:
        key = (entry["mode"], entry["path"])
        prior = first_seen.get(key)
        if prior is None:
            first_seen[key] = entry
            deduped.append(entry)
        elif prior["size"] is None and entry["size"] is not None:
            # Coalesce (#2148 round 3): a known size fills a prior unknown -
            # None means "the listing left the size unknown" (hub.py
            # list_repo_entries_complete is int | None), never a measurement
            # that can disagree.
            prior["size"] = entry["size"]
        elif (
            prior["size"] is not None
            and entry["size"] is not None
            and prior["size"] != entry["size"]
        ):
            return {
                "status": "ERROR",
                "url": "",
                "detail": (
                    f"row-index-duplicate-conflict: {entry['path']} listed more than "
                    f"once with CONFLICTING sizes ({prior['size']} vs {entry['size']}) "
                    "- overlapping --row-index-hf-prefix values re-listed the path "
                    "and the listings disagree; a real conflict is never silently "
                    "deduplicated (fail-loud, zero downloads performed)"
                ),
            }
        # Remaining arms (equal known sizes, or a later None against a known
        # value) are agreeing duplicates: count once, keep the known size.
    return deduped


def _attribute_row_index_entries(
    entries: list[dict],
    expected_rows: dict[str, int],
    exempt_labels: dict[str, str],
    glob_pattern: str,
) -> tuple[dict[str, list[dict]], dict | None]:
    """Attribute every matched file to EXACTLY ONE declared label (#2148).

    The three attribution ERROR arms (`row-index-missing`,
    `row-index-unattributed`, `row-index-label-ambiguous`) fire BEFORE any
    counting - a mis-scoped invocation must never shrink a denominator. A
    declared prefix must be label-COVERED; the remedy for unattributed /
    ambiguous files is per-label prefixes (the flag is repeatable) or
    disjoint label spellings, never a weaker arm.
    """
    labels = tuple(expected_rows)
    by_label: dict[str, list[dict]] = {label: [] for label in labels}
    problems: list[str] = []
    for entry in entries:
        matches = _labels_for_path(labels, entry["rel"])
        if not matches:
            problems.append(f"row-index-unattributed: {entry['rel']} matches no declared label")
        elif len(matches) > 1:
            problems.append(
                f"row-index-label-ambiguous: {entry['rel']} matches labels {sorted(matches)}"
            )
        else:
            by_label[matches[0]].append(entry)
    for label in labels:
        if not by_label[label] and label not in exempt_labels:
            problems.append(
                f"row-index-missing: declared label '{label}' matched zero index "
                f"files (glob {glob_pattern!r})"
            )
    if problems:
        return by_label, {
            "status": "ERROR",
            "url": "",
            "detail": (
                "; ".join(problems) + ". Attribution is fail-loud BEFORE any "
                "counting: every glob-matched file must resolve to exactly one "
                "declared label - scope per-label prefixes "
                "(--row-index-hf-prefix is repeatable) or use disjoint label "
                "spellings; a mis-scoped invocation is not a store defect (#2148)."
            ),
        }
    return by_label, None


def _row_index_budget_error(
    entries: list[dict],
    *,
    max_bytes: int,
    max_total_bytes: int,
    max_files: int,
    revision: str | None,
) -> dict | None:
    """Enforce the fetch budgets off the LISTING, before the first fetch.

    Unknown sizes are resolved with ONE batched retried ``get_paths_info``
    probe at the pinned ``revision``; a size that stays unknown is never
    assumed under cap (#2148). Every failing arm returns with ZERO downloads
    performed.
    """
    unknown = [e for e in entries if e["size"] is None]
    if unknown:
        paths = [e["path"] for e in unknown]
        try:
            resolved = _row_index_resolve_sizes(paths, revision=revision)
        except Exception as e:
            return {
                "status": "ERROR",
                "url": "",
                "detail": (
                    f"row-index-size-unknown: batched get_paths_info size probe "
                    f"failed for {len(paths)} file(s) ({type(e).__name__}: {e}); "
                    "an unprovable size is never assumed under cap (zero "
                    "downloads performed)"
                ),
            }
        for entry in unknown:
            entry["size"] = resolved.get(entry["path"])
        still = sorted(e["path"] for e in entries if e["size"] is None)
        if still:
            return {
                "status": "ERROR",
                "url": "",
                "detail": (
                    "row-index-size-unknown: size unresolved after the batched "
                    f"probe for: {', '.join(still)} - an unprovable size is "
                    "never assumed under cap (zero downloads performed)"
                ),
            }
    over = sorted((e["rel"], e["size"]) for e in entries if e["size"] > max_bytes)
    if over:
        described = ", ".join(f"{rel} ({size} B)" for rel, size in over)
        return {
            "status": "ERROR",
            "url": "",
            "detail": (
                f"row-index-file-over-cap: {described} exceeds "
                f"--row-index-max-bytes={max_bytes} (zero downloads performed)"
            ),
        }
    total = sum(e["size"] for e in entries)
    if len(entries) > max_files or total > max_total_bytes:
        return {
            "status": "ERROR",
            "url": "",
            "detail": (
                f"row-index-budget-exceeded: matched set is {len(entries)} "
                f"file(s) / {total} B against --row-index-max-files={max_files} "
                f"/ --row-index-max-total-bytes={max_total_bytes} (zero "
                "downloads performed)"
            ),
        }
    return None


def _count_row_index_file(
    text: str, key_fields: tuple[str, ...]
) -> tuple[int, set[tuple[str, ...]] | None, list[str]]:
    """Count one index file: (non-empty lines, distinct key tuples, violations).

    With no declared key the distinct set is None (line-count-only mode). A
    row missing a declared key field - or unparseable as JSON - is a
    `row-index-key-absent` violation, never silently skipped (a skipped row
    would shrink the very denominator this check gates on).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not key_fields:
        return len(lines), None, []
    distinct: set[tuple[str, ...]] = set()
    violations: list[str] = []
    for i, ln in enumerate(lines, start=1):
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            violations.append(f"line {i}: unparseable JSON ({e})")
            continue
        if not isinstance(obj, dict):
            violations.append(f"line {i}: JSON row is not an object")
            continue
        missing = [f for f in key_fields if f not in obj]
        if missing:
            violations.append(f"line {i}: missing key field(s) {missing}")
            continue
        distinct.add(tuple(json.dumps(obj[f], sort_keys=True) for f in key_fields))
    return len(lines), distinct, violations


def _read_row_index_entry(entry: dict, *, revision: str | None) -> str:
    """Read one budget-passed index file's text (local stat'd file, or a
    KB-scale Hub fetch at the pinned ``revision`` via the retried staging
    seam into a temp dir)."""
    if entry["mode"] == "local":
        return Path(entry["path"]).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="rowindex-") as td:
        target = Path(td) / entry["path"].rsplit("/", 1)[-1]
        fetched = _row_index_fetch(entry["path"], target, revision=revision)
        return Path(fetched).read_text(encoding="utf-8")


def _count_label_entries(
    by_label: dict[str, list[dict]], key_fields: tuple[str, ...], *, revision: str | None
) -> tuple[dict[str, dict], list[str]]:
    """Fetch + count every attributed index file, per label, at ``revision``.

    Returns ``(counts, errors)``: counts[label] = {lines, distinct, shards};
    errors collects fetch failures + key-absent violations (both fail-loud).
    Emits flushed ``[realized-rows] unit k/N`` progress lines to STDERR -
    unconditionally, never stdout: the canonical Step 2.11 invocation
    carries ``--json`` ("Output raw JSON"), so stdout must stay a single
    parseable document (#2148 round 3; detached runs redirect ``2>&1`` into
    one log, satisfying the code-style per-unit progress convention
    identically). One ``fetch-start`` line lands BEFORE each fetch so a
    hanging fetch names the file it is stuck on, plus one completion line
    after.
    """
    counts: dict[str, dict] = {}
    errors: list[str] = []
    total = sum(len(v) for v in by_label.values())
    done = 0
    t0 = time.monotonic()
    for label, label_entries in by_label.items():
        lines_total = 0
        distinct: set[tuple[str, ...]] = set()
        shards: list[str] = []
        for entry in sorted(label_entries, key=lambda e: str(e["rel"])):
            done += 1
            print(
                f"[realized-rows] unit {done}/{total} {entry['rel']} "
                f"fetch-start elapsed={time.monotonic() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            try:
                text = _read_row_index_entry(entry, revision=revision)
            except Exception as e:
                errors.append(
                    f"row-index fetch/read failed for {entry['rel']}: {type(e).__name__}: {e}"
                )
                print(
                    f"[realized-rows] unit {done}/{total} {entry['rel']} "
                    f"elapsed={time.monotonic() - t0:.1f}s (fetch/read FAILED)",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            n_lines, file_distinct, violations = _count_row_index_file(text, key_fields)
            errors.extend(f"row-index-key-absent: {entry['rel']}: {v}" for v in violations)
            lines_total += n_lines
            if file_distinct is not None:
                distinct |= file_distinct
            shards.append(f"{str(entry['rel']).rsplit('/', 1)[-1]}:{n_lines}")
            print(
                f"[realized-rows] unit {done}/{total} {entry['rel']} "
                f"elapsed={time.monotonic() - t0:.1f}s",
                file=sys.stderr,
                flush=True,
            )
        counts[label] = {
            "lines": lines_total,
            "distinct": len(distinct) if key_fields else None,
            "shards": shards,
        }
    return counts, errors


def _realized_rows_label_verdict(
    label: str,
    expected: int,
    counted: dict,
    key_fields: tuple[str, ...],
    self_reported_rows: dict[str, int],
    exempt_labels: dict[str, str],
) -> tuple[dict, str]:
    """Per-label verdict row + detail part (#2148 verdict lattice).

    Gate quantity: the DISTINCT count of the declared full-key tuple - never
    the raw line count (a healthy repaired store holds MORE lines than rows:
    #2091 post-repair is 2048 lines / 2000 distinct) and never a producer
    self-reported field (reported for context only; #2091's fields were
    wrong in BOTH directions). No key declared -> line count is a floor
    only: `lines >= expected` is a WARN (`realized-rows-no-distinct-key`),
    never an OK. Exempt labels WARN visibly with realized counts intact.
    """
    lines = counted["lines"]
    distinct = counted["distinct"]
    gate_quantity = distinct if key_fields else lines
    row: dict = {
        "expected": expected,
        "realized_lines": lines,
        "realized_distinct": distinct,
        "duplicates": (lines - distinct) if distinct is not None else None,
        "shards": counted["shards"],
        "key_fields": list(key_fields),
    }
    notes: list[str] = []
    if label in self_reported_rows:
        row["self_reported"] = self_reported_rows[label]
        notes.append(
            f"self-reported={self_reported_rows[label]} (producer self-reported, context only)"
        )
        if self_reported_rows[label] != gate_quantity:
            notes.append("producer-field-mismatch (reported, never gated)")
    if label in exempt_labels:
        verdict, tag = "WARN", "realized-rows-exempt"
        notes.append(f"exempt: {exempt_labels[label]}")
    elif key_fields:
        if distinct < expected:
            verdict, tag = "FAIL", "realized-rows-short"
        elif distinct > expected:
            verdict, tag = "FAIL", "realized-rows-unexpected-surplus"
        else:
            verdict, tag = "OK", ""
    elif lines < expected:
        verdict, tag = "FAIL", "realized-rows-short"
    else:
        verdict, tag = "WARN", "realized-rows-no-distinct-key"
        notes.append(
            "no --row-index-distinct-key declared: a line count >= expected is "
            "uninformative about within-file shortfall (declare the full row "
            "identity to resolve)"
        )
    row["verdict"] = verdict
    row["tag"] = tag
    part = (
        f"{label}: {verdict}"
        + (f" {tag}" if tag else "")
        + f" expected={expected}"
        + f" distinct={distinct if distinct is not None else 'n/a'}"
        + f" lines={lines}"
        + f" duplicates={row['duplicates'] if row['duplicates'] is not None else 'n/a'}"
        + f" shards=[{','.join(counted['shards'])}]"
        + f" key=({','.join(key_fields) if key_fields else 'none'})"
    )
    if notes:
        part += " [" + "; ".join(notes) + "]"
    return row, part


def _realized_rows_invocation_error(
    expected_rows: dict[str, int],
    hf_prefixes: tuple[str, ...],
    local_root: str | None,
    exempt_labels: dict[str, str],
) -> dict | None:
    """Entry-time INVOCATION validation for the realized row-count check
    (#2148 rounds 2-3): returns the ERROR row for a malformed invocation,
    or None when it is well-formed. Every arm runs BEFORE the
    no-expectation SKIP and before ANY Hub read (zero Hub reads performed),
    so the callable owes the same contract the CLI parser enforces - a
    direct caller cannot slip a reasonless/unmatched exemption, a
    dual-source invocation, or an empty prefix past an early SKIP return.
    """
    # (0) Exemption validation (#2148 round 2, concern
    # `exemption-validation-after-skip`). Flag-free legacy invocations
    # still SKIP at the caller.
    blank_reasons = sorted(lb for lb, reason in exempt_labels.items() if not str(reason).strip())
    if blank_reasons:
        return {
            "status": "ERROR",
            "url": "",
            "detail": (
                "realized-rows-exempt-invalid: an exemption reason is MANDATORY "
                f"and non-empty; label(s) with a blank reason: "
                f"{', '.join(blank_reasons)} (an exemption with no recorded "
                "reason is indistinguishable from a silenced shortfall)"
            ),
        }
    exempt_unmatched = sorted(set(exempt_labels) - set(expected_rows))
    if exempt_unmatched:
        return {
            "status": "ERROR",
            "url": "",
            "detail": (
                "realized-rows-exempt-unmatched: --realized-rows-exempt names "
                f"label(s) not in the declared --expected-rows set: "
                f"{', '.join(exempt_unmatched)} (rejects stale and typo'd "
                "exemptions; an exemption-only invocation ERRORs rather than "
                "SKIPping)"
            ),
        }

    # (0b) Source-shape validation (#2148 round 3). REFUSE, not union:
    # (i) the entry schema carries no mode-independent logical identity (a
    # local entry's `rel` is root-relative, an HF entry's `rel` is the full
    # repo path), so an honest cross-source union would need an identity
    # mapping the invocation does not carry; (ii) more fundamentally, a row
    # present only in the LOCAL source would satisfy --expected-rows on a
    # gate whose PASS licenses deleting that local copy - a local-only row
    # is precisely what is NOT durable.
    if hf_prefixes and local_root:
        return {
            "status": "ERROR",
            "url": "",
            "detail": (
                "row-index-dual-source: --row-index-local-root and "
                "--row-index-hf-prefix are mutually exclusive - pick ONE "
                "row-index source per invocation. No mode-independent row "
                "identity exists across the two sources, and a local-only "
                "row cannot prove Hub durability on a gate whose PASS "
                "licenses deleting the local copy (#2148 round 3; zero Hub "
                "reads performed)"
            ),
        }
    empty_prefixes = [repr(p) for p in hf_prefixes if not str(p).rstrip("/")]
    if empty_prefixes:
        return {
            "status": "ERROR",
            "url": "",
            "detail": (
                "row-index-prefix-empty: --row-index-hf-prefix canonicalizes "
                f"to an empty prefix ({', '.join(empty_prefixes)}) - a "
                "malformed flag is an invocation defect; silently skipping "
                "it would blame the store (row-index-missing) for the "
                "operator's flag (#2148 round 3; zero Hub reads performed)"
            ),
        }
    return None


def check_realized_row_counts(
    *,
    expected_rows: dict[str, int] | None = None,
    hf_prefixes: tuple[str, ...] = (),
    local_root: str | None = None,
    glob_pattern: str = ROW_INDEX_DEFAULT_GLOB,
    distinct_key_fields: tuple[str, ...] = (),
    self_reported_rows: dict[str, int] | None = None,
    exempt_labels: dict[str, str] | None = None,
    max_bytes: int = ROW_INDEX_MAX_BYTES_DEFAULT,
    max_total_bytes: int = ROW_INDEX_MAX_TOTAL_BYTES_DEFAULT,
    max_files: int = ROW_INDEX_MAX_FILES_DEFAULT,
) -> dict:
    """Reconcile REALIZED row-index contents vs the input-side declaration (#2148).

    The within-file sibling of :func:`check_outroot_residue`: the residue
    check binds DISK to the upload filters at FILE grain; this binds the
    CONTENT of present ``row_index*.jsonl`` files to the run's INPUT-side
    expectation. #2091 PASSed every file-level check while ~25% of rows were
    missing INSIDE present files, because the count check read the
    producer's self-reported ``capture_rows`` - literally
    ``manifest.get("n_rows")``, the expectation echoed back.

    Mechanics, in order: (0) validate the INVOCATION - exemption reasons
    non-empty AND labels member of the declared set (#2148 round 2), the
    two row-index sources MUTUALLY EXCLUSIVE (`row-index-dual-source`,
    #2148 round 3: no mode-independent row identity exists across the two
    entry namespaces, and a local-only row would satisfy an expectation on
    a gate whose PASS licenses deleting that local copy), and no
    empty-after-strip prefix (`row-index-prefix-empty`) - all BEFORE even
    the no-expectation SKIP and before ANY Hub read, so a malformed direct
    call ERRORs exactly as the CLI parser rejects it; (1) resolve ONE
    pinned Hub revision when any HF prefix is declared (retried
    ``repo_info(...).sha``, the ``stage_hub_prefix`` pattern) - the listing
    walks, the size probe, and every staged fetch read that SAME SHA,
    reported in the check detail; (2) enumerate the declared source (one
    scoped tree walk per DISTINCT prefix / one local walk), deduplicating
    matched entries by (mode, path) - overlapping prefixes count a file
    ONCE, an unknown (None) size coalesces with a known one, and a
    same-path duplicate with CONFLICTING KNOWN sizes ERRORs; (3) attribute
    every matched file to EXACTLY ONE declared label - the three
    attribution ERROR arms fire BEFORE any counting; (4) enforce the
    per-file AND aggregate fetch budgets off the deduplicated listing (ONE
    batched retried ``get_paths_info`` probe for unknown sizes; every
    failing arm returns with ZERO downloads); (5) fetch + count non-empty
    lines and distinct declared-key tuples (per-file progress rides STDERR
    - stdout stays a single parseable document under ``--json``); (6)
    apply the LABEL-grained exemptions (always a visible WARN row that
    still reports realized counts); (7) per-label verdict lattice +
    ERROR > FAIL > WARN > OK reduction. SKIP fires only when no
    expectation is declared, so legacy invocations gain exactly one inert
    SKIP row.
    """
    expected_rows = dict(expected_rows or {})
    self_reported_rows = dict(self_reported_rows or {})
    exempt_labels = dict(exempt_labels or {})

    # (0)+(0b) Invocation validation FIRST - before even the no-expectation
    # SKIP and before ANY Hub read (#2148 rounds 2-3): exemption reasons +
    # membership, source mutual exclusion, no empty-after-strip prefix.
    invocation_error = _realized_rows_invocation_error(
        expected_rows, tuple(hf_prefixes), local_root, exempt_labels
    )
    if invocation_error is not None:
        return invocation_error

    if not expected_rows:
        extra = (
            " (a row-index source was supplied without an expectation)"
            if (hf_prefixes or local_root)
            else ""
        )
        return {
            "status": "SKIP",
            "url": "",
            "detail": ("no --expected-rows LABEL=N declared; realized row counts not checked")
            + extra,
        }
    if not hf_prefixes and not local_root:
        return {
            "status": "ERROR",
            "url": "",
            "detail": (
                "--expected-rows declared but no row-index source supplied "
                "(--row-index-hf-prefix / --row-index-local-root) - a missing "
                "input never reads as a clean zero (fail-loud)"
            ),
        }

    # (1) ONE pinned revision for every Hub read this verdict performs
    # (#2148 round 2, blocker `row-index-moving-head-snapshot` + plan §4(A)
    # step 3). Local-only invocations perform no Hub read and pin nothing.
    revision: str | None = None
    if hf_prefixes:
        try:
            revision = _row_index_resolve_revision()
        except Exception as e:
            return {
                "status": "ERROR",
                "url": "",
                "detail": (
                    f"row-index revision resolve failed: {type(e).__name__}: {e} "
                    "- every Hub read (listing / size probe / fetch) pins ONE "
                    "revision; an unpinnable snapshot is never read (fail-loud, "
                    "zero listing/probe/fetch reads performed - the failed "
                    "repo_info was itself the only Hub read)"
                ),
            }

    entries = _row_index_entries(tuple(hf_prefixes), local_root, glob_pattern, revision=revision)
    if isinstance(entries, dict):
        return entries

    by_label, attribution_error = _attribute_row_index_entries(
        entries, expected_rows, exempt_labels, glob_pattern
    )
    if attribution_error is not None:
        return attribution_error

    budget_error = _row_index_budget_error(
        entries,
        max_bytes=max_bytes,
        max_total_bytes=max_total_bytes,
        max_files=max_files,
        revision=revision,
    )
    if budget_error is not None:
        return budget_error

    key_fields = tuple(distinct_key_fields or ())
    label_counts, count_errors = _count_label_entries(by_label, key_fields, revision=revision)
    if count_errors:
        return {
            "status": "ERROR",
            "url": "",
            "detail": (
                "; ".join(count_errors) + ". A row missing its declared key - or an "
                "unreadable index file - is never silently skipped: skipping would "
                "shrink the very denominator this check gates on (#2148)."
            ),
        }

    labels_report: dict[str, dict] = {}
    statuses: list[str] = []
    parts: list[str] = []
    for label in sorted(expected_rows):
        counted = label_counts.get(label) or {
            "lines": 0,
            "distinct": 0 if key_fields else None,
            "shards": [],
        }
        row, part = _realized_rows_label_verdict(
            label, expected_rows[label], counted, key_fields, self_reported_rows, exempt_labels
        )
        labels_report[label] = row
        statuses.append(row["verdict"])
        parts.append(part)

    if any(s == "FAIL" for s in statuses):
        status = "FAIL"
    elif any(s == "WARN" for s in statuses):
        status = "WARN"
    else:
        status = "OK"
    result = {
        "status": status,
        "url": "",
        "detail": (
            "; ".join(parts) + " | gate quantity: distinct full-key rows, never a "
            "raw line count and never a producer-reported field (#2148)"
        ),
        "labels": labels_report,
    }
    if revision is not None:
        # Traceability (#2148 round 2): the verdict names the ONE Hub
        # snapshot every listing / probe / fetch read.
        result["revision"] = revision
        result["detail"] += f" | hub revision: {revision}"
    return result


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
    outroot_listing: str | None = None,
    outroot: str | None = None,
    hf_prefixes: tuple[str, ...] = (),
    outroot_exempt: tuple[str, ...] = (),
    discarded_names: tuple[str, ...] = (),
    expected_rows: dict[str, int] | None = None,
    row_index_hf_prefixes: tuple[str, ...] = (),
    row_index_local_root: str | None = None,
    row_index_glob: str = ROW_INDEX_DEFAULT_GLOB,
    row_index_distinct_key: tuple[str, ...] = (),
    self_reported_rows: dict[str, int] | None = None,
    realized_rows_exempt: dict[str, str] | None = None,
    row_index_max_bytes: int = ROW_INDEX_MAX_BYTES_DEFAULT,
    row_index_max_total_bytes: int = ROW_INDEX_MAX_TOTAL_BYTES_DEFAULT,
    row_index_max_files: int = ROW_INDEX_MAX_FILES_DEFAULT,
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
        wandb_prose_note: str | None = None
        if card_check is None:
            # The card declared no structured wandb field; a free-text
            # line under the bare ``wandb`` key is consulted by NO check
            # (#612 follow-up) — remember it so whichever fallback row
            # lands below names the unconsulted declaration. Append-only:
            # row status and the convention-probe fallback are unchanged.
            wandb_prose_note = _unconsulted_prose_wandb_note(results_card)
            # Nothing declared anywhere (#608 follow-up): probe the
            # conventional <default_entity>/issue<N> project for
            # issue<N>_* runs before hard-MISSING. Fail-soft — a probe
            # error keeps the strict MISSING row below.
            card_check = check_wandb_runs_convention_project(issue_num)
        wandb_row = card_check or {
            "status": "MISSING",
            "url": "",
            "detail": (
                "No WandB run path provided (no epm:results "
                "reproducibility_card declares wandb_run_path / "
                "wandb_run_names, and the conventional-project probe "
                f"found no issue{issue_num}_* runs)"
            ),
        }
        if wandb_prose_note:
            wandb_row["detail"] = f"{wandb_row['detail']}; ALSO: {wandb_prose_note}"
        report["checks"]["wandb_run"] = wandb_row

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

    # 9. Out-root residue (#2187): name-set diff of the run's out-root files
    # vs the union of HF prefixes + ISSUE-SCOPED git trees + declared
    # discards. SKIPs (inert) when the caller supplies no listing, so legacy
    # invocations differ only by this one SKIP row.
    report["checks"]["outroot_residue"] = check_outroot_residue(
        issue_num,
        outroot_listing=outroot_listing,
        outroot=outroot,
        hf_prefixes=tuple(hf_prefixes or ()),
        exempt_globs=tuple(outroot_exempt or ()),
        discarded_names=tuple(discarded_names or ()),
    )

    # 10. Realized row-count reconciliation (#2148): distinct full-key rows
    # REALLY inside the store's row_index*.jsonl files vs the input-side
    # declaration - never a producer self-reported count (#2091 echoed the
    # expectation back and PASSed a ~25% shortfall). SKIPs (inert) when no
    # --expected-rows is declared, so legacy invocations differ only by
    # this one SKIP row.
    report["checks"]["realized_row_counts"] = check_realized_row_counts(
        expected_rows=expected_rows,
        hf_prefixes=tuple(row_index_hf_prefixes or ()),
        local_root=row_index_local_root,
        glob_pattern=row_index_glob,
        distinct_key_fields=tuple(row_index_distinct_key or ()),
        self_reported_rows=self_reported_rows,
        exempt_labels=realized_rows_exempt,
        max_bytes=row_index_max_bytes,
        max_total_bytes=row_index_max_total_bytes,
        max_files=row_index_max_files,
    )

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


def _parse_label_count_args(
    parser: argparse.ArgumentParser, pairs: list[str], flag: str
) -> dict[str, int]:
    """Parse repeatable ``LABEL=N`` flags into a dict (fail-loud on malformed
    / duplicate / negative input - a silently-dropped expectation would
    un-gate the very label it was declared for, #2148)."""
    out: dict[str, int] = {}
    for raw in pairs:
        label, sep, value = str(raw).partition("=")
        if not sep or not label:
            parser.error(f"{flag} expects LABEL=N, got {raw!r}")
        if label in out:
            parser.error(f"{flag} declares label {label!r} twice")
        try:
            count = int(value)
        except ValueError:
            parser.error(f"{flag} expects an integer count, got {raw!r}")
        if count < 0:
            parser.error(f"{flag} expects a non-negative count, got {raw!r}")
        out[label] = count
    return out


def _parse_label_reason_args(
    parser: argparse.ArgumentParser, pairs: list[str], flag: str
) -> dict[str, str]:
    """Parse repeatable ``LABEL=REASON`` flags (reason MANDATORY and
    non-empty: an exemption with no recorded reason is indistinguishable
    from a silenced shortfall, #2148)."""
    out: dict[str, str] = {}
    for raw in pairs:
        label, sep, reason = str(raw).partition("=")
        if not sep or not label or not reason.strip():
            parser.error(f"{flag} expects LABEL=REASON with a non-empty reason, got {raw!r}")
        if label in out:
            parser.error(f"{flag} declares label {label!r} twice")
        out[label] = reason.strip()
    return out


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
    parser.add_argument(
        "--outroot-listing",
        help=(
            "Path to a file with one absolute out-root file path per line "
            "(the pod-side `find <out-root> -type f | sort` capture). Feeds "
            "the #2187 out-root residue check: every listed file must match "
            "a permanent home (HF prefix / issue-scoped git tree) or a "
            "declared discard, by NAME-SET diff — never by count."
        ),
    )
    parser.add_argument(
        "--outroot",
        help=(
            "Local out-root directory to walk instead of --outroot-listing "
            "(same #2187 residue check; mutually complementary inputs)"
        ),
    )
    parser.add_argument(
        "--hf-prefix",
        action="append",
        default=[],
        dest="hf_prefixes",
        help=(
            "HF data-repo prefix the run wrote (repeatable; issue-scoped by "
            "construction, e.g. issueN_<slug>/raw_completions). Enumerate "
            "EVERY prefix the run wrote — a missing prefix makes its files "
            "read as residue (fail-toward-FAIL)."
        ),
    )
    parser.add_argument(
        "--outroot-exempt",
        action="append",
        default=[],
        help="fnmatch glob exempted from the out-root residue diff (repeatable)",
    )
    parser.add_argument(
        "--discarded-name",
        action="append",
        default=[],
        dest="discarded_names",
        help=(
            "Basename covered by a plan §10 discarded_artifacts entry "
            "(repeatable; text/JSON is never discardable)"
        ),
    )
    parser.add_argument(
        "--expected-rows",
        action="append",
        default=[],
        metavar="LABEL=N",
        help=(
            "Input-side expected row count per artifact label (repeatable). "
            "Arms the #2148 realized row-count reconciliation: the gate "
            "quantity is the DISTINCT count of --row-index-distinct-key "
            "tuples REALLY inside the label's row_index*.jsonl files - never "
            "a raw line count, never a producer-reported field."
        ),
    )
    parser.add_argument(
        "--row-index-hf-prefix",
        action="append",
        default=[],
        dest="row_index_hf_prefixes",
        help=(
            "HF data-repo prefix holding one label's row-index files "
            "(repeatable; scope PER LABEL - a whole-store prefix whose files "
            "match no declared label trips row-index-unattributed by design)"
        ),
    )
    parser.add_argument(
        "--row-index-local-root",
        help=(
            "Local directory to walk for row-index files INSTEAD OF "
            "--row-index-hf-prefix (mutually exclusive; e.g. staged "
            "pre-teardown copies). A local-only row cannot prove Hub "
            "durability, so a mixed-source invocation is refused (#2148)."
        ),
    )
    parser.add_argument(
        "--row-index-glob",
        default=ROW_INDEX_DEFAULT_GLOB,
        help=f"fnmatch basename glob selecting index files (default {ROW_INDEX_DEFAULT_GLOB!r})",
    )
    parser.add_argument(
        "--row-index-distinct-key",
        default="",
        metavar="FIELD[,FIELD...]",
        help=(
            "Comma-separated JSON fields forming the FULL logical row "
            "identity (unit key + any draw/rollout index). Omitted: the line "
            "count is a floor only - lines >= expected WARNs "
            "(realized-rows-no-distinct-key), never OK."
        ),
    )
    parser.add_argument(
        "--self-reported-rows",
        action="append",
        default=[],
        metavar="LABEL=N",
        help=(
            "Producer self-reported count per label (repeatable; REPORTED "
            "for context, never gated on - #2091's fields were wrong in "
            "both directions)"
        ),
    )
    parser.add_argument(
        "--realized-rows-exempt",
        action="append",
        default=[],
        metavar="LABEL=REASON",
        help=(
            "Exempt one declared label from the FAIL lattice with a "
            "MANDATORY reason (repeatable). The label still counts and "
            "emits a visible WARN row with its realized figures; an "
            "exemption naming an undeclared label ERRORs."
        ),
    )
    parser.add_argument(
        "--row-index-max-bytes",
        type=int,
        default=ROW_INDEX_MAX_BYTES_DEFAULT,
        help="Per-file fetch cap in bytes (over-cap ERRORs with zero downloads)",
    )
    parser.add_argument(
        "--row-index-max-total-bytes",
        type=int,
        default=ROW_INDEX_MAX_TOTAL_BYTES_DEFAULT,
        help="Aggregate fetch cap in bytes (over-cap ERRORs with zero downloads)",
    )
    parser.add_argument(
        "--row-index-max-files",
        type=int,
        default=ROW_INDEX_MAX_FILES_DEFAULT,
        help="Matched-set file-count cap (over-cap ERRORs with zero downloads)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--no-fail", action="store_true", help="Don't exit with error on FAIL")

    args = parser.parse_args()

    if args.row_index_hf_prefixes and args.row_index_local_root:
        parser.error(
            "--row-index-local-root and --row-index-hf-prefix are mutually "
            "exclusive: pick ONE row-index source per invocation (#2148 - no "
            "cross-source row identity exists, and a local-only row cannot "
            "prove Hub durability)"
        )

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
        outroot_listing=args.outroot_listing,
        outroot=args.outroot,
        hf_prefixes=tuple(args.hf_prefixes),
        outroot_exempt=tuple(args.outroot_exempt),
        discarded_names=tuple(args.discarded_names),
        expected_rows=_parse_label_count_args(parser, args.expected_rows, "--expected-rows"),
        row_index_hf_prefixes=tuple(args.row_index_hf_prefixes),
        row_index_local_root=args.row_index_local_root,
        row_index_glob=args.row_index_glob,
        row_index_distinct_key=tuple(
            f.strip() for f in args.row_index_distinct_key.split(",") if f.strip()
        ),
        self_reported_rows=_parse_label_count_args(
            parser, args.self_reported_rows, "--self-reported-rows"
        ),
        realized_rows_exempt=_parse_label_reason_args(
            parser, args.realized_rows_exempt, "--realized-rows-exempt"
        ),
        row_index_max_bytes=args.row_index_max_bytes,
        row_index_max_total_bytes=args.row_index_max_total_bytes,
        row_index_max_files=args.row_index_max_files,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))

    if report["verdict"] == "FAIL" and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
