"""Issue #2658 unit 9 — the C0-C5 comparator ladder (plan section 6).

Fits and scores the six comparator classes on the frozen layer-19
answer-span-mean vectors (plan section 5), under the development-only fitting
discipline (plan section 9) and content-superfamily grouped folds:

- C0  nuisance logistic (prompt/response length, source-frame / stratum
      indicators, seed position, truncation flag — NO prompt identity);
- C1  dev-only text baselines (prompt-text / answer-text / combined-text;
      frozen TF-IDF fit on dev fold-train + pinned all-MiniLM-L6-v2);
- C2  exact frozen external direction dot product, UNCALIBRATED, with the
      PREREGISTERED sign (never sign selection — a flipped sign is a failure);
- C3  dev-calibrated scalar logistic on the C2 score;
- C4  dev-only raw class-mean direction (dev labels) + scalar logistic;
- C5  full L2 logistic probe on the 3584-dim vector after dev-only
      per-dimension z-scoring.

C3-C5 estimator (plan section 6, registered in ``issue2658_common.ESTIMATOR``):
sklearn ``LogisticRegression(solver="lbfgs", max_iter=5000, tol=1e-7)``, no
class weighting, C grid 10^[-6..4], FIVE content-superfamily folds, selection
by macro conditional AUROC (mean over folds of the fold's equal-prompt macro
within-prompt AUROC), tie-break to the STRONGEST regularization within 1e-4.
Pooled cross-fold predictions never feed a confirmatory metric — the
``FoldPredictions`` container keeps fold arrays per-fold and its ``pooled()``
accessor RAISES (plan section 9 fail-on list).

Rows without a hash-verifiable frozen external direction emit
``not-estimable — no frozen external direction`` for C2/C3 (never a proxy
direction); the partition is READ from the committed provenance artifact
(``eval_results/issue_2658/direction_provenance.json``), not re-derived.

UNIT-10 SEAM (permutation tests / Holm families / hierarchical bootstrap are
unit 10's, deliberately NOT here): unit 10 imports ``load_comparator_results``
+ ``load_unit_scores`` (per-response TEST scores + labels + prompt/superfamily
ids per (row, comparator), plus the selected-C per-fold dev validation arrays
— per-fold, never pooled), the ``NOT_ESTIMABLE`` records, and
``macro_selection_metric`` / ``FoldPredictions`` for any dev-side reads.

Checkpoint grain: one ledger unit per (row, comparator, fold) plus one
refit+test unit per (row, comparator) — > 50 units at the full ladder, so
per-unit atomic JSONL persistence + parameter-keyed resume + one flushed
progress line per unit (code-style checkpoint-per-phase, #823/#1689). Resume
keys hash GENERATING PARAMETERS only (never recomputed float bytes, #1336).

CONTENT HYGIENE: prompt/answer text flows disk -> memory -> vectorizer only;
logs, ledger records, and score files carry ids, counts, and hashes — never
row text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy/sklearn import

import numpy as np  # noqa: E402
from scipy import sparse as sp  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2658_capture as CAP  # noqa: E402
import issue2658_common as C  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_judge as J  # noqa: E402
import issue2658_objective_labels as OL  # noqa: E402
import issue2658_power as PW  # noqa: E402  (equal_prompt_macro_auroc — never a second AUROC)
from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

# ---------------------------------------------------------------------------
# Registered constants.
# ---------------------------------------------------------------------------
COMP_SCHEMA = "i2658-comparator-unit-v1"
SCORES_SCHEMA = "i2658-comparator-scores-v1"
LOGIC_VERSION = "i2658-u9-logic-v1"  # bump on any output-affecting logic change

# Ladder order (plan section 6). C1 realizes three text surfaces.
COMPARATORS = (
    "c0_nuisance",
    "c1_text_prompt",
    "c1_text_answer",
    "c1_text_combined",
    "c2_direction_dot",
    "c3_direction_calibrated",
    "c4_devmean_calibrated",
    "c5_full_probe",
)
DIRECTION_COMPARATORS = ("c2_direction_dot", "c3_direction_calibrated")
FITTED_COMPARATORS = tuple(c for c in COMPARATORS if c != "c2_direction_dot")

# == issue2658_provenance.NOT_ESTIMABLE (equality pinned by tests; the local
# literal avoids a module-top torch import for every consumer).
NOT_ESTIMABLE = "not-estimable — no frozen external direction"

# C grid walked ASCENDING = strongest regularization first (warm-start walk).
C_GRID: tuple[float, ...] = tuple(sorted(float(c) for c in C.ESTIMATOR["c_grid"]))
C_GRID_GENERATING = ("pow10", -6, 4)  # machine-stable resume-key form (#1336)
TIE_TOL = 1e-4
N_FOLDS = int(C.ESTIMATOR["n_folds"])
MIN_USABLE_FOLDS = 3

MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_MAX_FEATURES = 4096

_ESTIMATOR_KEY = {k: v for k, v in C.ESTIMATOR.items() if k != "c_grid"}


# ---------------------------------------------------------------------------
# Errors (plan section 9 fail-on list; base class from issue2658_common).
# ---------------------------------------------------------------------------
class ComparatorInputError(C.Issue2658GuardError):
    """Malformed / incomplete comparator inputs."""


class UnderdeterminedFitError(C.Issue2658GuardError):
    """n_train < d full-probe fit refused without a recorded justification."""


class SignConventionError(C.Issue2658GuardError):
    """Frozen-direction sign/hash contract violated (never sign selection)."""


class FoldConstructionError(C.Issue2658GuardError):
    """Content-superfamily folds could not be constructed as registered."""


# ---------------------------------------------------------------------------
# Data model.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResponseRow:
    """One retained answer: identity, grouping, label, nuisance fields, text.

    ``prompt_len_tokens`` / ``answer_len_tokens`` come from the gen record
    (``n_prompt_tokens`` / ``n_completion_tokens``); ``truncated`` is the gen
    record's ``finish_reason == "length"``; ``seed_position`` is the draw slot
    (``response_index``). Text fields are consumed by C1 only and are NEVER
    printed or persisted by this module.
    """

    prompt_id: str
    response_index: int
    superfamily_id: str
    split: str  # "dev" | "test" (content-superfamily data split)
    label: bool
    prompt_len_tokens: int
    answer_len_tokens: int
    source_frame: str
    stratum: str
    seed_position: int
    truncated: bool
    prompt_text: str
    answer_text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.prompt_id, self.response_index)


@dataclass
class RowData:
    """One row's assembled dataset: X aligned with rows, dev+test present."""

    row: str
    X: np.ndarray  # (n, d) float32 C-contiguous
    rows: list[ResponseRow]
    data_fingerprint: str
    label_source: str  # "judge-cells" | "objective-labels" | "synthetic"
    synthetic_direction: np.ndarray | None = None  # tests/smoke C2 seam only

    def __post_init__(self) -> None:
        self.X = np.ascontiguousarray(self.X, dtype=np.float32)
        if self.X.ndim != 2 or self.X.shape[0] != len(self.rows) or not len(self.rows):
            raise ComparatorInputError(
                f"row {self.row}: X {self.X.shape} misaligned with {len(self.rows)} rows"
            )
        if not np.isfinite(self.X).all():
            raise ComparatorInputError(f"row {self.row}: non-finite activation values")
        keys = [r.key for r in self.rows]
        if len(set(keys)) != len(keys):
            raise ComparatorInputError(f"row {self.row}: duplicate (prompt_id, response_index)")
        splits = {r.split for r in self.rows}
        if not splits <= {"dev", "test"}:
            raise ComparatorInputError(f"row {self.row}: unexpected splits {sorted(splits)}")
        if splits != {"dev", "test"}:
            raise ComparatorInputError(
                f"row {self.row}: both dev and test rows are required, got {sorted(splits)}"
            )
        sf_by_split: dict[str, set[str]] = {"dev": set(), "test": set()}
        for r in self.rows:
            sf_by_split[r.split].add(r.superfamily_id)
        C.assert_split_lineage_disjoint(sf_by_split["dev"], sf_by_split["test"])
        # One prompt never spans two superfamilies or two splits.
        seen: dict[str, tuple[str, str]] = {}
        for r in self.rows:
            prior = seen.setdefault(r.prompt_id, (r.superfamily_id, r.split))
            if prior != (r.superfamily_id, r.split):
                raise ComparatorInputError(
                    f"row {self.row}: prompt {r.prompt_id} carries inconsistent "
                    "superfamily/split assignments"
                )

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    def mask(self, split: str) -> np.ndarray:
        return np.array([r.split == split for r in self.rows], dtype=bool)


def join_labels(
    keys: Sequence[tuple[str, int]], label_records: dict[tuple[str, int], dict[str, Any]]
) -> tuple[dict[tuple[str, int], bool], dict[str, int]]:
    """Join label records onto capture keys. A key with NO record raises
    ``MissingLabelError`` (silent omission is a plan-section-9 fail-on); a
    record whose status is a named non-scored/non-labeled class is EXCLUDED
    with a per-status count (drop-never-coerce)."""
    str_keys = [f"{pid}#r{idx:02d}" for pid, idx in keys]
    C.assert_labels_complete(
        str_keys,
        {f"{pid}#r{idx:02d}": label_records.get((pid, idx)) for pid, idx in keys},
    )
    kept: dict[tuple[str, int], bool] = {}
    excluded: dict[str, int] = {}
    for k in keys:
        rec = label_records[k]
        status = str(rec["status"])
        if status in ("scored", "labeled"):
            if not isinstance(rec["label"], bool):
                raise C.CoercedLabelError(
                    f"label record for {k} has status {status!r} but non-bool label "
                    f"{rec['label']!r}; drop-never-coerce"
                )
            kept[k] = rec["label"]
        else:
            excluded[status] = excluded.get(status, 0) + 1
    return kept, excluded


# ---------------------------------------------------------------------------
# Content-superfamily folds (grouped; a superfamily is never split).
# ---------------------------------------------------------------------------
def superfamily_folds(dev_rows: Sequence[ResponseRow], n_folds: int = N_FOLDS) -> dict[str, int]:
    """Deterministic balanced fold assignment PER SUPERFAMILY (largest-first
    greedy over row counts; ties broken by sorted superfamily id, then lowest
    fold index). Grouping by superfamily makes splitting one structurally
    impossible. Machine-stable: integer counts + string ids only (#1946)."""
    counts: dict[str, int] = {}
    for r in dev_rows:
        counts[r.superfamily_id] = counts.get(r.superfamily_id, 0) + 1
    if len(counts) < n_folds:
        raise FoldConstructionError(
            f"{len(counts)} dev superfamilies < n_folds={n_folds}; the registered "
            "content-superfamily folds cannot be constructed"
        )
    assignment: dict[str, int] = {}
    load = [0] * n_folds
    for sf in sorted(counts, key=lambda s: (-counts[s], s)):
        f = min(range(n_folds), key=lambda i: (load[i], i))
        assignment[sf] = f
        load[f] += counts[sf]
    return assignment


def usable_folds(
    dev_rows: Sequence[ResponseRow], assignment: dict[str, int], n_folds: int = N_FOLDS
) -> tuple[list[int], dict[str, Any]]:
    """Folds whose validation slice realizes >= 1 discordant prompt.

    Discordance is LABEL-structural (score-independent): a prompt counts iff
    its validation labels realize both classes. Degenerate folds are excluded
    WITH a counted record; < MIN_USABLE_FOLDS usable folds raises."""
    by_fold_prompt: dict[tuple[int, str], set[bool]] = {}
    for r in dev_rows:
        by_fold_prompt.setdefault((assignment[r.superfamily_id], r.prompt_id), set()).add(r.label)
    usable = [
        f
        for f in range(n_folds)
        if any(fold == f and len(v) == 2 for (fold, _), v in by_fold_prompt.items())
    ]
    report = {
        "n_folds": n_folds,
        "usable_folds": usable,
        "n_degenerate_folds": n_folds - len(usable),
    }
    if len(usable) < MIN_USABLE_FOLDS:
        raise ComparatorInputError(
            f"only {len(usable)} of {n_folds} folds realize a discordant validation prompt "
            f"(< {MIN_USABLE_FOLDS}); selection metric undefined — refusing"
        )
    return usable, report


# ---------------------------------------------------------------------------
# Per-fold predictions: per-fold BY CONSTRUCTION; pooled access RAISES.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FoldPredictions:
    """Per-fold validation predictions. There is deliberately NO pooled
    accessor: ``pooled()`` raises, and the only confirmatory scorer in this
    module (:func:`macro_selection_metric`) consumes per-fold arrays."""

    fold_ids: tuple[int, ...]
    scores: tuple[np.ndarray, ...]
    labels: tuple[np.ndarray, ...]
    prompt_ids: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        n = len(self.fold_ids)
        if not (len(self.scores) == len(self.labels) == len(self.prompt_ids) == n and n):
            raise ComparatorInputError("FoldPredictions arrays misaligned or empty")

    def pooled(self) -> None:
        """Pooled cross-fold predictions are structurally unavailable."""
        raise C.PooledFoldMetricError(
            "pooled cross-fold predictions requested from FoldPredictions; plan "
            "section 6/9 forbids pooled-fold confirmatory metrics — consume per-fold arrays"
        )


def macro_selection_metric(fp: FoldPredictions) -> float:
    """Macro conditional AUROC: the unweighted mean over folds of the fold's
    equal-prompt macro within-prompt AUROC (unit 8's registered metric)."""
    C.assert_not_pooled_fold("per-fold")
    vals: list[float] = []
    for s, lab, pid in zip(fp.scores, fp.labels, fp.prompt_ids, strict=True):
        macro, n_disc = PW.equal_prompt_macro_auroc(
            np.asarray(s, dtype=np.float64), np.asarray(lab).astype(bool), np.asarray(pid)
        )
        if int(n_disc) > 0:
            vals.append(float(macro))
    if not vals:
        raise ComparatorInputError("no fold realized a discordant prompt; metric undefined")
    return float(np.mean(vals))


def select_c(per_c_macro: dict[float, float]) -> float:
    """Best macro conditional AUROC; ties within TIE_TOL of the best go to the
    STRONGEST regularization (smallest C) — plan section 6 tie-break."""
    if set(per_c_macro) != set(C_GRID):
        raise ComparatorInputError(
            f"selection table covers {sorted(per_c_macro)} != registered grid {list(C_GRID)}"
        )
    best = max(per_c_macro.values())
    if not np.isfinite(best):
        raise ComparatorInputError("selection metric non-finite for every C")
    return min(c for c, m in per_c_macro.items() if best - m <= TIE_TOL)


# ---------------------------------------------------------------------------
# Dev-only transform guards + feature pipes.
# ---------------------------------------------------------------------------
def _assert_dev_fit_rows(rows: Sequence[ResponseRow], what: str) -> None:
    """Every learned transform/calibration/probe fits on dev rows only."""
    C.assert_transform_fit_split("dev")
    bad = sorted({r.split for r in rows if r.split != "dev"})
    if bad:
        raise C.TestDerivedTransformError(
            f"{what}: fit rows include non-dev splits {bad} (plan section 9: "
            "fail on test-derived transforms)"
        )
    if not rows:
        raise ComparatorInputError(f"{what}: empty fit row set")


@dataclass
class ZScore:
    """Per-dimension z-scoring, statistics from dev fold-train rows ONLY."""

    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, rows: Sequence[ResponseRow], scope: str = "dev-fold-train"):
        C.assert_preprocessing_scope(scope)
        _assert_dev_fit_rows(rows, "ZScore.fit")
        mean = X.astype(np.float64).mean(axis=0)
        sd = X.astype(np.float64).std(axis=0, ddof=0)
        sd = np.where(sd > 0, sd, 1.0)  # constant dims center to 0
        return cls(mean=mean, sd=sd)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X.astype(np.float64) - self.mean) / self.sd


class NuisanceFeaturizer:
    """C0 features. Consumes ONLY nuisance fields — the transform reads
    (prompt_len_tokens, answer_len_tokens, seed_position, truncated,
    source_frame, stratum); prompt identity is structurally invisible."""

    def __init__(self) -> None:
        self.frames: tuple[str, ...] | None = None
        self.strata: tuple[str, ...] | None = None

    def fit(self, rows: Sequence[ResponseRow], scope: str = "dev-fold-train"):
        C.assert_preprocessing_scope(scope)
        _assert_dev_fit_rows(rows, "NuisanceFeaturizer.fit")
        self.frames = tuple(sorted({r.source_frame for r in rows}))
        self.strata = tuple(sorted({r.stratum for r in rows}))
        return self

    def transform(self, rows: Sequence[ResponseRow]) -> np.ndarray:
        if self.frames is None or self.strata is None:
            raise ComparatorInputError("NuisanceFeaturizer.transform before fit")
        fpos = {f: i for i, f in enumerate(self.frames)}
        spos = {s: i for i, s in enumerate(self.strata)}
        n = len(rows)
        out = np.zeros((n, 4 + len(fpos) + len(spos)), dtype=np.float64)
        for i, r in enumerate(rows):
            out[i, 0] = np.log1p(r.prompt_len_tokens)
            out[i, 1] = np.log1p(r.answer_len_tokens)
            out[i, 2] = float(r.seed_position)
            out[i, 3] = 1.0 if r.truncated else 0.0
            j = fpos.get(r.source_frame)
            if j is not None:
                out[i, 4 + j] = 1.0
            k = spos.get(r.stratum)
            if k is not None:
                out[i, 4 + len(fpos) + k] = 1.0
        return out


class HashEmbedBackend:
    """Deterministic offline embedding substitute (smoke/tests ONLY): 64 dims
    from sha256 of the text. Enumerated as a smoke blind-spot — the production
    MiniLM import/download path is NOT executed when this backend is active."""

    name = "hash64"
    dim = 64

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode()).digest()
            raw = np.frombuffer(
                hashlib.sha512(h).digest() + hashlib.sha512(h[::-1]).digest(), dtype=np.uint8
            )
            out[i] = (raw[: self.dim].astype(np.float32) - 127.5) / 127.5
        return out


class MiniLMBackend:
    """Pinned all-MiniLM-L6-v2 sentence embeddings (production C1 backend).

    Revision pin: freeze-on-first-use — the first production run resolves the
    model repo's current main commit sha and persists it to
    ``<out_root>/comparators/minilm_pin.json``; every later run REQUIRES the
    stored pin (stale/mismatched pin raises via check_cache_entry semantics).
    """

    name = "all-MiniLM-L6-v2"

    def __init__(self, pin_path: Path) -> None:
        self.pin_path = pin_path
        self._model = None
        self.revision = self._resolve_pin()

    def _resolve_pin(self) -> str:
        if self.pin_path.exists():
            pin = json.loads(self.pin_path.read_text())
            if pin.get("model_id") != MINILM_MODEL_ID or not pin.get("revision"):
                raise C.CacheStaleError(
                    f"minilm pin at {self.pin_path} is malformed or pins a different "
                    f"model ({pin.get('model_id')!r}); refusing to substitute"
                )
            return str(pin["revision"])
        from huggingface_hub import HfApi

        sha = HfApi().model_info(MINILM_MODEL_ID).sha
        if not sha:
            raise ComparatorInputError(f"could not resolve a revision for {MINILM_MODEL_ID}")
        self.pin_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.pin_path,
            {"model_id": MINILM_MODEL_ID, "revision": sha, "frozen": "on-first-use"},
        )
        print(f"[comparators] minilm pin frozen-on-first-use: {sha}", flush=True)
        return sha

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MINILM_MODEL_ID, revision=self.revision)
        vecs = self._model.encode(
            list(texts), batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(vecs, dtype=np.float32)


class TextFeaturizer:
    """C1: frozen TF-IDF (vocab+idf fit on dev fold-train ONLY) concatenated
    with pinned sentence embeddings (a frozen transform — no fitting)."""

    KINDS = ("prompt", "answer", "combined")

    def __init__(self, kind: str, backend: Any) -> None:
        if kind not in self.KINDS:
            raise ComparatorInputError(f"unknown text kind {kind!r}")
        self.kind = kind
        self.backend = backend
        self.tfidf: TfidfVectorizer | None = None
        # Embedding cache lives on the SHARED backend (embeddings are a frozen
        # pinned transform): one encode per unique text per process, amortized
        # across every fold/refit unit instead of re-encoding per unit.
        if not hasattr(backend, "_text_cache"):
            backend._text_cache = {}
        self._embed_cache: dict[str, np.ndarray] = backend._text_cache

    def _texts(self, rows: Sequence[ResponseRow]) -> list[str]:
        if self.kind == "prompt":
            return [r.prompt_text for r in rows]
        if self.kind == "answer":
            return [r.answer_text for r in rows]
        return [r.prompt_text + "\n\n" + r.answer_text for r in rows]

    def fit(self, rows: Sequence[ResponseRow], scope: str = "dev-fold-train"):
        C.assert_preprocessing_scope(scope)
        _assert_dev_fit_rows(rows, f"TextFeaturizer[{self.kind}].fit")
        self.tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, sublinear_tf=True)
        self.tfidf.fit(self._texts(rows))
        return self

    def _embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._embed_cache]
        if missing:
            vecs = self.backend.encode(missing)
            for t, v in zip(missing, vecs, strict=True):
                self._embed_cache[t] = v
        return np.stack([self._embed_cache[t] for t in texts], axis=0)

    def transform(self, rows: Sequence[ResponseRow]):
        if self.tfidf is None:
            raise ComparatorInputError("TextFeaturizer.transform before fit")
        texts = self._texts(rows)
        return sp.hstack(
            [self.tfidf.transform(texts), sp.csr_matrix(self._embed(texts))], format="csr"
        )


def devmean_direction(
    X: np.ndarray, rows: Sequence[ResponseRow], y: np.ndarray, scope: str = "dev-fold-train"
) -> np.ndarray:
    """C4 direction: raw class-mean difference computed on dev fold-train
    labels ONLY (positive mean minus negative mean)."""
    C.assert_preprocessing_scope(scope)
    _assert_dev_fit_rows(rows, "devmean_direction")
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        raise ComparatorInputError("devmean_direction: fit rows realize a single class")
    Xf = X.astype(np.float64)
    return (Xf[y].mean(axis=0) - Xf[~y].mean(axis=0)).astype(np.float64)


def assert_probe_wellposed(
    n_train: int, d: int, allow_underdetermined: str | None
) -> dict[str, Any]:
    """C5 well-posedness gate: refuse n_train < d without a RECORDED
    justification (deliberately under-determined regimes must be explicit)."""
    under = n_train < d
    if under and not allow_underdetermined:
        raise UnderdeterminedFitError(
            f"full-probe fit refused: n_train={n_train} < d={d} with no recorded "
            "justification (pass --allow-underdetermined '<why>' for a deliberate "
            "under-determined regime, e.g. a smoke shape)"
        )
    return {
        "n_train": int(n_train),
        "d": int(d),
        "underdetermined": bool(under),
        "justification": allow_underdetermined if under else None,
    }


# ---------------------------------------------------------------------------
# Frozen-direction provenance seam (C2/C3).
# ---------------------------------------------------------------------------
def load_committed_provenance(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else F.PROVENANCE_PATH
    if not p.exists():
        raise ComparatorInputError(
            f"committed direction provenance not found at {p}; unit 9 reads the "
            "partition from the committed artifact, never re-derives it"
        )
    return json.loads(p.read_text())


def c2c3_partition(prov: dict[str, Any]) -> dict[str, list[str]]:
    part = prov.get("c2_c3_partition")
    if not part or set(part.get("eligible", [])) | set(part.get("not_estimable", [])) != set(
        C.ROW_IDS
    ):
        raise ComparatorInputError(
            "committed c2_c3_partition missing or does not cover the 11 registered rows"
        )
    return {"eligible": list(part["eligible"]), "not_estimable": list(part["not_estimable"])}


def verify_direction_entry(entry: dict[str, Any], vec: np.ndarray) -> np.ndarray:
    """Preregistered-sign + committed-hash contract for a loaded direction.

    The committed sign convention is '+dot = construct-positive' for every
    eligible row; the direction is applied AS COMMITTED (sign +1). Any hash
    mismatch or non-'+dot' convention RAISES — never a proxy, never a flip."""
    if entry.get("c2_c3") != "eligible":
        raise SignConventionError(
            f"row {entry.get('row')!r} is not C2/C3-eligible in the committed partition"
        )
    conv = str(entry.get("sign_convention", ""))
    if not conv.startswith("+dot"):
        raise SignConventionError(
            f"row {entry.get('row')!r} sign convention {conv!r} is not the preregistered "
            "'+dot = construct-positive' form; sign selection is forbidden"
        )
    vec = np.ascontiguousarray(np.asarray(vec, dtype=np.float32))
    if vec.shape != (C.HIDDEN,) or not np.isfinite(vec).all():
        raise SignConventionError(f"direction shape/finite check failed: {vec.shape}")
    got = hashlib.sha256(vec.tobytes()).hexdigest()  # provenance vector_sha256 domain
    if got != entry.get("vector_sha256"):
        raise SignConventionError(
            f"loaded direction sha {got} != committed {entry.get('vector_sha256')} for row "
            f"{entry.get('row')!r}; refusing (never a substituted/proxy direction)"
        )
    return vec


def production_direction_provider(row: str, prov: dict[str, Any]) -> np.ndarray:
    """Load the frozen external direction for an eligible row through the
    provenance module's OWN loaders, then verify against the COMMITTED entry
    (hash pin makes any loader drift fail loud)."""
    import issue2658_provenance as P  # deferred: torch + plot3 loaders

    entry = {e["row"]: e for e in prov["rows"]}[row]
    spec = {s.row: s for s in P.ROWS}[row]
    if spec.kind == "rb28":
        vec = P.P3._load_rb28(P.resolve_file(spec.rel), C.LAYER)
    elif spec.kind == "axis":
        vec = P.P3._load_axis(P.resolve_file(spec.rel), C.LAYER)
    elif spec.kind == "refusal_diff":
        labels_path = P.resolve_file("issue2356_refusalpred/armA/labels.json")
        rows_rel = "issue2356_refusalpred/analysis_tensors/consolidated/armA.rows.json"
        vc_rel = "issue2356_refusalpred/analysis_tensors/consolidated/armA__v_C.npy"
        stage_root = P._common_stage_root(
            {rows_rel: P.resolve_file(rows_rel), vc_rel: P.resolve_file(vc_rel)}
        )
        P.P3.LABELS_2356 = labels_path
        vec, _counts = P.P3._refusal_direction(stage_root, C.LAYER)
    elif spec.kind == "correctness":
        vec, _prov = P._load_correctness(P.resolve_file(spec.rel), spec.surface)
    else:
        raise SignConventionError(f"row {row!r} has no frozen-direction loader (kind={spec.kind})")
    return verify_direction_entry(entry, vec)


# ---------------------------------------------------------------------------
# Estimator core (plan section 6 wire instrument).
# ---------------------------------------------------------------------------
def _fresh_estimator() -> LogisticRegression:
    # penalty is left at its default (l2) — sklearn 1.8 deprecates passing the
    # kwarg explicitly; the registered ESTIMATOR["penalty"] == "l2" is asserted.
    if C.ESTIMATOR["penalty"] != "l2":
        raise ComparatorInputError("registered estimator penalty drifted from l2")
    return LogisticRegression(
        solver=str(C.ESTIMATOR["solver"]),
        max_iter=int(C.ESTIMATOR["max_iter"]),
        tol=float(C.ESTIMATOR["tol"]),
        class_weight=C.ESTIMATOR["class_weight"],  # None — no class weighting
        warm_start=True,
        C=C_GRID[0],
    )


def fit_c_grid_warm(
    X_train: Any, y_train: np.ndarray, X_val: Any
) -> dict[float, tuple[np.ndarray, int]]:
    """Walk the registered C grid strong -> weak regularization with ONE
    warm-started estimator; return per-C validation decision scores + n_iter."""
    y = np.asarray(y_train, dtype=bool)
    if y.all() or not y.any():
        raise ComparatorInputError("fit_c_grid_warm: training rows realize a single class")
    est = _fresh_estimator()
    out: dict[float, tuple[np.ndarray, int]] = {}
    for c in C_GRID:  # ascending C = strongest regularization first
        est.C = float(c)
        est.fit(X_train, y)
        out[c] = (
            est.decision_function(X_val).astype(np.float32),
            int(np.max(est.n_iter_)),
        )
    return out


def refit_at_c(X_train: Any, y_train: np.ndarray, c: float) -> LogisticRegression:
    est = _fresh_estimator()
    est.warm_start = False
    est.C = float(c)
    est.fit(X_train, np.asarray(y_train, dtype=bool))
    return est


# ---------------------------------------------------------------------------
# Ledger (mirrors unit 8's PowerLedger; per-unit atomic append + resume).
# ---------------------------------------------------------------------------
class CompLedger:
    """Atomic-append JSONL keyed on GENERATING PARAMETERS (never recomputed
    float bytes, #1336). Single-writer; flush+fsync per unit."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._done: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("schema") != COMP_SCHEMA:
                        raise ComparatorInputError(
                            f"{self.path}: foreign schema {rec.get('schema')!r} in ledger"
                        )
                    self._done[rec["key"]] = rec

    @staticmethod
    def unit_key(
        *,
        row: str,
        comparator: str,
        fold: str,
        data_fingerprint: str,
        label_source: str,
        direction_sha256: str,
        preprocessing: str,
        seed: int,
        embed_backend: str,
    ) -> str:
        base = C.cache_key(
            inputs_sha256=data_fingerprint,
            direction_sha256=direction_sha256,
            split="dev-fit/test-score",
            judge_fingerprint=label_source,
            estimator=_ESTIMATOR_KEY,
            grid=list(C_GRID_GENERATING),
            preprocessing=preprocessing,
            code_sha=LOGIC_VERSION,
            container="machine-independent",
            seeds={"seed": int(seed), "embed_backend": embed_backend},
        )
        body = json.dumps(
            {"cache_key": base, "row": row, "comparator": comparator, "fold": fold},
            sort_keys=True,
        )
        return hashlib.sha256(body.encode()).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        return self._done.get(key)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._done[record["key"]] = record


@dataclass
class _UnitCounter:
    done: int = 0
    cap: int = 0
    t0: float = field(default_factory=time.time)
    resumed: int = 0
    first_unit_wall: dict[str, float] = field(default_factory=dict)


def _npz_atomic(path: Path, **arrays: np.ndarray) -> str:
    """Atomic .npz write (tmp named <stem>.tmp.npz — np.savez appends .npz to
    names not ending in it, #1092); returns the file sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _progress(counter: _UnitCounter, row: str, comp: str, fold: str, note: str) -> None:
    print(
        f"[comparators] unit {counter.done}/{counter.cap} row={row} comp={comp} "
        f"fold={fold} elapsed={time.time() - counter.t0:.1f}s{note}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Feature pipes per comparator.
# ---------------------------------------------------------------------------
class _Pipe:
    """fit(rows, X, y) on dev fold-train; transform(rows, X) -> features."""

    preprocessing = "none"

    def fit(self, rows: Sequence[ResponseRow], X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, rows: Sequence[ResponseRow], X: np.ndarray) -> Any:
        raise NotImplementedError


class _C0Pipe(_Pipe):
    preprocessing = "c0:nuisance-vocab@dev-fold-train"

    def __init__(self) -> None:
        self.f = NuisanceFeaturizer()

    def fit(self, rows, X, y) -> None:
        self.f.fit(rows, scope="dev-fold-train")

    def transform(self, rows, X):
        return self.f.transform(rows)


class _C1Pipe(_Pipe):
    def __init__(self, kind: str, backend: Any) -> None:
        self.t = TextFeaturizer(kind, backend)
        self.preprocessing = f"c1:{kind}:tfidf@dev-fold-train+embed[{backend.name}]"

    def fit(self, rows, X, y) -> None:
        self.t.fit(rows, scope="dev-fold-train")

    def transform(self, rows, X):
        return self.t.transform(rows)


class _C3Pipe(_Pipe):
    """Scalar feature = frozen-direction dot; only the logistic is learned."""

    preprocessing = "c3:frozen-direction-dot (no fitted transform)"

    def __init__(self, direction: np.ndarray) -> None:
        self.v = np.asarray(direction, dtype=np.float64)

    def fit(self, rows, X, y) -> None:
        return None  # nothing learned; the calibration logistic is dev-guarded downstream

    def transform(self, rows, X):
        return (X.astype(np.float64) @ self.v)[:, None]


class _C4Pipe(_Pipe):
    preprocessing = "c4:class-mean-direction@dev-fold-train"

    def __init__(self) -> None:
        self.w: np.ndarray | None = None

    def fit(self, rows, X, y) -> None:
        self.w = devmean_direction(X, rows, y, scope="dev-fold-train")

    def transform(self, rows, X):
        if self.w is None:
            raise ComparatorInputError("_C4Pipe.transform before fit")
        return (X.astype(np.float64) @ self.w)[:, None]


class _C5Pipe(_Pipe):
    preprocessing = "c5:zscore@dev-fold-train"

    def __init__(self) -> None:
        self.z: ZScore | None = None

    def fit(self, rows, X, y) -> None:
        self.z = ZScore.fit(X, rows, scope="dev-fold-train")

    def transform(self, rows, X):
        if self.z is None:
            raise ComparatorInputError("_C5Pipe.transform before fit")
        return self.z.transform(X)


def make_pipe(comp: str, embed_backend: Any, direction: np.ndarray | None) -> _Pipe:
    if comp == "c0_nuisance":
        return _C0Pipe()
    if comp.startswith("c1_text_"):
        return _C1Pipe(comp.removeprefix("c1_text_"), embed_backend)
    if comp == "c3_direction_calibrated":
        if direction is None:
            raise ComparatorInputError("c3 requires a frozen direction")
        return _C3Pipe(direction)
    if comp == "c4_devmean_calibrated":
        return _C4Pipe()
    if comp == "c5_full_probe":
        return _C5Pipe()
    raise ComparatorInputError(f"no feature pipe for comparator {comp!r}")


# ---------------------------------------------------------------------------
# Unit runners.
# ---------------------------------------------------------------------------
@dataclass
class LadderContext:
    rowdata: RowData
    ledger: CompLedger
    counter: _UnitCounter
    scores_dir: Path
    embed_backend: Any
    direction: np.ndarray | None
    direction_sha: str
    seed: int
    allow_underdetermined: str | None
    fold_assignment: dict[str, int]
    usable: list[int]
    fold_report: dict[str, Any]

    def key_for(self, comp: str, fold: str, preprocessing: str) -> str:
        return CompLedger.unit_key(
            row=self.rowdata.row,
            comparator=comp,
            fold=fold,
            data_fingerprint=self.rowdata.data_fingerprint,
            label_source=self.rowdata.label_source,
            direction_sha256=self.direction_sha,
            preprocessing=preprocessing,
            seed=self.seed,
            embed_backend=getattr(self.embed_backend, "name", "n/a"),
        )


def _dev_arrays(ctx: LadderContext) -> tuple[list[ResponseRow], np.ndarray, np.ndarray]:
    mask = ctx.rowdata.mask("dev")
    rows = [r for r in ctx.rowdata.rows if r.split == "dev"]
    return rows, ctx.rowdata.X[mask], np.array([r.label for r in rows], dtype=bool)


def _test_arrays(ctx: LadderContext) -> tuple[list[ResponseRow], np.ndarray, np.ndarray]:
    mask = ctx.rowdata.mask("test")
    rows = [r for r in ctx.rowdata.rows if r.split == "test"]
    return rows, ctx.rowdata.X[mask], np.array([r.label for r in rows], dtype=bool)


def _fold_npz_path(ctx: LadderContext, comp: str, fold: int) -> Path:
    return ctx.scores_dir / f"{ctx.rowdata.row}__{comp}__fold{fold}.npz"


def _final_npz_path(ctx: LadderContext, comp: str) -> Path:
    return ctx.scores_dir / f"{ctx.rowdata.row}__{comp}.npz"


def run_fold_unit(ctx: LadderContext, comp: str, fold: int) -> dict[str, Any]:
    """One (row, comparator, fold) unit: fit the pipe + walk the C grid on the
    fold-train slice, score the fold-validation slice, persist per-C val
    scores + macros. Resumable by generating-parameter key."""
    pipe = make_pipe(comp, ctx.embed_backend, ctx.direction)
    key = ctx.key_for(comp, f"fold{fold}", pipe.preprocessing)
    ctx.counter.done += 1
    prior = ctx.ledger.get(key)
    npz_path = _fold_npz_path(ctx, comp, fold)
    if prior is not None:
        if not npz_path.exists():
            raise C.CacheStaleError(
                f"ledger has fold unit {key[:12]}... but {npz_path} is missing; "
                "quarantine the ledger or the scores dir before resuming"
            )
        ctx.counter.resumed += 1
        _progress(ctx.counter, ctx.rowdata.row, comp, f"fold{fold}", " resume-skip")
        return prior
    t0 = time.time()
    dev_rows, X_dev, y_dev = _dev_arrays(ctx)
    tr_idx = [i for i, r in enumerate(dev_rows) if ctx.fold_assignment[r.superfamily_id] != fold]
    va_idx = [i for i, r in enumerate(dev_rows) if ctx.fold_assignment[r.superfamily_id] == fold]
    tr_rows = [dev_rows[i] for i in tr_idx]
    va_rows = [dev_rows[i] for i in va_idx]
    if not tr_idx or not va_idx:
        raise FoldConstructionError(f"fold {fold}: empty train or validation slice")
    Xtr_raw, Xva_raw = X_dev[tr_idx], X_dev[va_idx]
    pipe.fit(tr_rows, Xtr_raw, y_dev[tr_idx])
    Ftr = pipe.transform(tr_rows, Xtr_raw)
    Fva = pipe.transform(va_rows, Xva_raw)
    wellposed = None
    if comp == "c5_full_probe":
        wellposed = assert_probe_wellposed(len(tr_idx), Ftr.shape[1], ctx.allow_underdetermined)
    grid = fit_c_grid_warm(Ftr, y_dev[tr_idx], Fva)
    va_labels = y_dev[va_idx]
    va_pids = np.array([r.prompt_id for r in va_rows])
    per_c_macro: dict[str, float] = {}
    for c, (scores, _n_iter) in grid.items():
        macro, _n_disc = PW.equal_prompt_macro_auroc(scores.astype(np.float64), va_labels, va_pids)
        per_c_macro[repr(c)] = float(macro)
    scores_mat = np.stack([grid[c][0] for c in C_GRID], axis=0)  # (n_C, n_val)
    sha = _npz_atomic(
        npz_path,
        schema=np.array(SCORES_SCHEMA),
        c_grid=np.array(C_GRID, dtype=np.float64),
        val_scores_per_c=scores_mat,
        val_labels=va_labels,
        val_prompt_ids=va_pids,
        val_superfamily_ids=np.array([r.superfamily_id for r in va_rows]),
        val_response_index=np.array([r.response_index for r in va_rows], dtype=np.int64),
    )
    rec = {
        "schema": COMP_SCHEMA,
        "key": key,
        "unit": "fold",
        "row": ctx.rowdata.row,
        "comparator": comp,
        "fold": fold,
        "n_train": len(tr_idx),
        "n_val": len(va_idx),
        "n_feature": int(Ftr.shape[1]),
        "per_c_macro": per_c_macro,
        "n_iter_per_c": {repr(c): grid[c][1] for c in C_GRID},
        "wellposed": wellposed,
        "preprocessing": pipe.preprocessing,
        "scores_file": str(npz_path),
        "scores_sha256": sha,
        "elapsed_s": time.time() - t0,
    }
    ctx.ledger.append(rec)
    _progress(ctx.counter, ctx.rowdata.row, comp, f"fold{fold}", f" (unit={rec['elapsed_s']:.1f}s)")
    fam = comp.split("_")[0]
    if fam not in ctx.counter.first_unit_wall:
        ctx.counter.first_unit_wall[fam] = rec["elapsed_s"]
        remaining = max(0, ctx.counter.cap - ctx.counter.done)
        print(
            f"[comparators] ops arithmetic ({fam}): measured one-unit wall "
            f"{rec['elapsed_s']:.1f}s x <= {remaining} remaining units -> projected "
            f"<= {rec['elapsed_s'] * remaining / 3600:.2f} h at this family's per-unit cost "
            f"({len(C_GRID)} warm-start fits/unit, n_train={len(tr_idx)}, "
            f"d={int(Ftr.shape[1])})",
            flush=True,
        )
    return rec


def run_refit_unit(ctx: LadderContext, comp: str, fold_records: list[dict[str, Any]]):
    """Selection (macro conditional AUROC over usable folds, strongest-reg
    tie-break) + dev refit + TEST scoring for one fitted comparator."""
    pipe = make_pipe(comp, ctx.embed_backend, ctx.direction)
    key = ctx.key_for(comp, "refit_test", pipe.preprocessing)
    ctx.counter.done += 1
    prior = ctx.ledger.get(key)
    if prior is not None:
        if not _final_npz_path(ctx, comp).exists():
            raise C.CacheStaleError(
                f"ledger has refit unit {key[:12]}... but the final scores file is missing"
            )
        ctx.counter.resumed += 1
        _progress(ctx.counter, ctx.rowdata.row, comp, "refit_test", " resume-skip")
        return prior
    t0 = time.time()
    per_c: dict[float, list[float]] = {c: [] for c in C_GRID}
    for rec in fold_records:
        if rec["fold"] not in ctx.usable:
            continue
        for c in C_GRID:
            per_c[c].append(float(rec["per_c_macro"][repr(c)]))
    per_c_macro = {c: float(np.mean(v)) for c, v in per_c.items()}
    selected = select_c(per_c_macro)
    # Selected-C per-fold validation predictions (per-fold, never pooled).
    fp = load_fold_predictions(ctx, comp, selected)
    dev_selection_macro = macro_selection_metric(fp)
    dev_rows, X_dev, y_dev = _dev_arrays(ctx)
    # Terminal refit: dev IS the training fold of the test-scoring model; the
    # registered scope literal applies (test is the held-out fold).
    pipe.fit(dev_rows, X_dev, y_dev)
    Fdev = pipe.transform(dev_rows, X_dev)
    wellposed = None
    if comp == "c5_full_probe":
        wellposed = assert_probe_wellposed(len(dev_rows), Fdev.shape[1], ctx.allow_underdetermined)
    est = refit_at_c(Fdev, y_dev, selected)
    te_rows, X_te, y_te = _test_arrays(ctx)
    te_scores = est.decision_function(pipe.transform(te_rows, X_te)).astype(np.float32)
    test_macro, test_disc = PW.equal_prompt_macro_auroc(
        te_scores.astype(np.float64), y_te, np.array([r.prompt_id for r in te_rows])
    )
    arrays: dict[str, np.ndarray] = {
        "schema": np.array(SCORES_SCHEMA),
        "selected_c": np.array([selected], dtype=np.float64),
        "test_scores": te_scores,
        "test_labels": y_te,
        "test_prompt_ids": np.array([r.prompt_id for r in te_rows]),
        "test_superfamily_ids": np.array([r.superfamily_id for r in te_rows]),
        "test_response_index": np.array([r.response_index for r in te_rows], dtype=np.int64),
        "fold_ids": np.array(list(fp.fold_ids), dtype=np.int64),
    }
    for i, f in enumerate(fp.fold_ids):
        arrays[f"fold{f}_scores"] = fp.scores[i]
        arrays[f"fold{f}_labels"] = fp.labels[i]
        arrays[f"fold{f}_prompt_ids"] = fp.prompt_ids[i]
    sha = _npz_atomic(_final_npz_path(ctx, comp), **arrays)
    rec = {
        "schema": COMP_SCHEMA,
        "key": key,
        "unit": "refit_test",
        "row": ctx.rowdata.row,
        "comparator": comp,
        "fold": "refit_test",
        "status": "scored",
        "selected_c": selected,
        "selection_table": {repr(c): per_c_macro[c] for c in C_GRID},
        "tie_break": C.ESTIMATOR["tie_break"],
        "dev_selection_macro": dev_selection_macro,
        "test_macro_auroc_descriptive": float(test_macro),
        "test_n_discordant_prompts": int(test_disc),
        "n_dev": len(dev_rows),
        "n_test": len(te_rows),
        "fold_report": ctx.fold_report,
        "wellposed": wellposed,
        "preprocessing": pipe.preprocessing,
        "scores_file": str(_final_npz_path(ctx, comp)),
        "scores_sha256": sha,
        "elapsed_s": time.time() - t0,
        "metadata": as_metadata_dict(git_provenance(), phase="comparators"),
    }
    ctx.ledger.append(rec)
    _progress(
        ctx.counter,
        ctx.rowdata.row,
        comp,
        "refit_test",
        f" (selected_c={selected:g} test_macro={float(test_macro):.4f})",
    )
    return rec


def load_fold_predictions(ctx: LadderContext, comp: str, selected_c: float) -> FoldPredictions:
    """Selected-C per-fold validation predictions from the persisted fold npzs."""
    c_pos = list(C_GRID).index(float(selected_c))
    fold_ids, scores, labels, pids = [], [], [], []
    for f in ctx.usable:
        with np.load(_fold_npz_path(ctx, comp, f), allow_pickle=False) as z:
            if str(z["schema"]) != SCORES_SCHEMA:
                raise C.CacheStaleError(f"foreign scores schema in fold npz for fold {f}")
            fold_ids.append(f)
            scores.append(z["val_scores_per_c"][c_pos].copy())
            labels.append(z["val_labels"].copy())
            pids.append(z["val_prompt_ids"].copy())
    return FoldPredictions(
        fold_ids=tuple(fold_ids),
        scores=tuple(scores),
        labels=tuple(labels),
        prompt_ids=tuple(pids),
    )


def run_c2_unit(ctx: LadderContext) -> dict[str, Any]:
    """C2: uncalibrated frozen-direction dot product with the preregistered
    sign (+1 as committed). No fit, no folds — scores only."""
    comp = "c2_direction_dot"
    key = ctx.key_for(comp, "score", "c2:none (uncalibrated dot)")
    ctx.counter.done += 1
    prior = ctx.ledger.get(key)
    if prior is not None:
        ctx.counter.resumed += 1
        _progress(ctx.counter, ctx.rowdata.row, comp, "score", " resume-skip")
        return prior
    t0 = time.time()
    if ctx.direction is None:
        raise SignConventionError(f"row {ctx.rowdata.row}: c2 requested with no direction")
    v = np.asarray(ctx.direction, dtype=np.float64)
    dev_rows, X_dev, y_dev = _dev_arrays(ctx)
    te_rows, X_te, y_te = _test_arrays(ctx)
    te_scores = (X_te.astype(np.float64) @ v).astype(np.float32)
    dev_scores = (X_dev.astype(np.float64) @ v).astype(np.float32)
    test_macro, test_disc = PW.equal_prompt_macro_auroc(
        te_scores.astype(np.float64), y_te, np.array([r.prompt_id for r in te_rows])
    )
    sha = _npz_atomic(
        _final_npz_path(ctx, comp),
        schema=np.array(SCORES_SCHEMA),
        test_scores=te_scores,
        test_labels=y_te,
        test_prompt_ids=np.array([r.prompt_id for r in te_rows]),
        test_superfamily_ids=np.array([r.superfamily_id for r in te_rows]),
        test_response_index=np.array([r.response_index for r in te_rows], dtype=np.int64),
        dev_scores=dev_scores,
        dev_labels=y_dev,
        dev_prompt_ids=np.array([r.prompt_id for r in dev_rows]),
        dev_fold_of_row=np.array(
            [ctx.fold_assignment[r.superfamily_id] for r in dev_rows], dtype=np.int64
        ),
    )
    rec = {
        "schema": COMP_SCHEMA,
        "key": key,
        "unit": "c2_score",
        "row": ctx.rowdata.row,
        "comparator": comp,
        "fold": "score",
        "status": "scored",
        "sign": "+1 (preregistered '+dot = construct-positive'; no sign selection)",
        "direction_sha256": ctx.direction_sha,
        "test_macro_auroc_descriptive": float(test_macro),
        "test_n_discordant_prompts": int(test_disc),
        "n_dev": len(dev_rows),
        "n_test": len(te_rows),
        "scores_file": str(_final_npz_path(ctx, comp)),
        "scores_sha256": sha,
        "elapsed_s": time.time() - t0,
        "metadata": as_metadata_dict(git_provenance(), phase="comparators"),
    }
    ctx.ledger.append(rec)
    _progress(ctx.counter, ctx.rowdata.row, comp, "score", " (uncalibrated)")
    return rec


def emit_not_estimable(ctx: LadderContext, comp: str) -> dict[str, Any]:
    key = ctx.key_for(comp, "not_estimable", "n/a")
    ctx.counter.done += 1
    prior = ctx.ledger.get(key)
    if prior is not None:
        ctx.counter.resumed += 1
        _progress(ctx.counter, ctx.rowdata.row, comp, "not_estimable", " resume-skip")
        return prior
    rec = {
        "schema": COMP_SCHEMA,
        "key": key,
        "unit": "not_estimable",
        "row": ctx.rowdata.row,
        "comparator": comp,
        "fold": "not_estimable",
        "status": "not-estimable",
        "reason": NOT_ESTIMABLE,
        "elapsed_s": 0.0,
    }
    ctx.ledger.append(rec)
    _progress(ctx.counter, ctx.rowdata.row, comp, "not_estimable", f" ({NOT_ESTIMABLE})")
    return rec


def run_ladder(
    rowdata: RowData,
    comps: Sequence[str],
    *,
    ledger: CompLedger,
    counter: _UnitCounter,
    scores_dir: Path,
    embed_backend: Any,
    partition: dict[str, list[str]],
    direction_provider: Callable[[str], np.ndarray] | None,
    seed: int,
    allow_underdetermined: str | None,
) -> list[dict[str, Any]]:
    """Run the requested comparators for one row; returns final records."""
    dev_rows = [r for r in rowdata.rows if r.split == "dev"]
    assignment = superfamily_folds(dev_rows, N_FOLDS)
    usable, fold_report = usable_folds(dev_rows, assignment, N_FOLDS)
    not_estimable = rowdata.row in partition["not_estimable"]
    direction = None
    direction_sha = "n/a"
    if any(c in DIRECTION_COMPARATORS for c in comps) and not not_estimable:
        if direction_provider is None:
            raise SignConventionError(
                f"row {rowdata.row} is C2/C3-eligible but no direction provider was supplied"
            )
        direction = np.ascontiguousarray(direction_provider(rowdata.row), dtype=np.float32)
        if direction.shape != (rowdata.d,):
            raise SignConventionError(
                f"direction shape {direction.shape} != (d={rowdata.d},) for row {rowdata.row}"
            )
        direction_sha = hashlib.sha256(direction.tobytes()).hexdigest()
    ctx = LadderContext(
        rowdata=rowdata,
        ledger=ledger,
        counter=counter,
        scores_dir=scores_dir,
        embed_backend=embed_backend,
        direction=direction,
        direction_sha=direction_sha,
        seed=seed,
        allow_underdetermined=allow_underdetermined,
        fold_assignment=assignment,
        usable=usable,
        fold_report=fold_report,
    )
    out: list[dict[str, Any]] = []
    for comp in comps:
        if comp in DIRECTION_COMPARATORS and not_estimable:
            out.append(emit_not_estimable(ctx, comp))
            continue
        if comp == "c2_direction_dot":
            out.append(run_c2_unit(ctx))
            continue
        fold_records = [run_fold_unit(ctx, comp, f) for f in range(N_FOLDS)]
        out.append(run_refit_unit(ctx, comp, fold_records))
    return out


def units_for(row: str, comps: Sequence[str], partition: dict[str, list[str]]) -> int:
    n = 0
    for comp in comps:
        if comp in DIRECTION_COMPARATORS and row in partition["not_estimable"]:
            n += 1
        elif comp == "c2_direction_dot":
            n += 1
        else:
            n += N_FOLDS + 1
    return n


# ---------------------------------------------------------------------------
# Unit-10 seam: result loaders.
# ---------------------------------------------------------------------------
def load_comparator_results(comparators_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Final (row, comparator) records from the ledger: refit_test / c2_score /
    not_estimable units only (fold units are internal)."""
    ledger = CompLedger(Path(comparators_dir) / "ledger.jsonl")
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in ledger._done.values():
        if rec["unit"] in ("refit_test", "c2_score", "not_estimable"):
            out[(rec["row"], rec["comparator"])] = rec
    return out


def load_unit_scores(record: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load + hash-verify one final record's scores file (unit 10 entry)."""
    if record.get("status") == "not-estimable":
        raise ComparatorInputError(
            f"{record['row']}/{record['comparator']} is {NOT_ESTIMABLE}; no scores exist"
        )
    path = Path(record["scores_file"])
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != record["scores_sha256"]:
        raise C.CacheStaleError(f"scores file {path} sha {got} != ledger {record['scores_sha256']}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


# ---------------------------------------------------------------------------
# Production data assembly (dev+test artifacts of the frozen pipeline).
# ---------------------------------------------------------------------------
def _split_assignment_for_row(row: str) -> dict[str, str]:
    m = json.loads(F.SPLIT_MANIFEST_PATH.read_text())
    entries = [r for r in m["rows"] if r["row"] == row]
    if len(entries) != 1:
        raise ComparatorInputError(f"split manifest carries {len(entries)} entries for {row!r}")
    return dict(entries[0]["superfamily_splits"])


def _read_manifest_rows(out_root: Path, split: str, row: str) -> dict[tuple[str, int], dict]:
    d = out_root / "gen_manifest" / split
    files = sorted(d.glob(f"{row}__*.jsonl"))
    if not files:
        raise ComparatorInputError(f"no gen manifest files for row {row!r} under {d}")
    out: dict[tuple[str, int], dict] = {}
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                m = json.loads(line)
                if m["row"] != row or m["split"] != split:
                    raise ComparatorInputError(f"{path.name}: foreign manifest row {m['row']!r}")
                out[(m["prompt_id"], int(m["response_index"]))] = m
    return out


def _read_gen_records(out_root: Path, split: str, row: str) -> dict[tuple[str, int], dict]:
    d = out_root / "raw_completions" / split
    files = sorted(d.glob(f"{row}__*.json"))
    if not files:
        raise ComparatorInputError(f"no gen cell files for row {row!r} under {d}")
    out: dict[tuple[str, int], dict] = {}
    for path in files:
        body = json.loads(path.read_text())
        if body.get("schema") != G.GEN_SCHEMA:
            raise ComparatorInputError(f"{path.name}: gen schema {body.get('schema')!r}")
        for r in body["records"]:
            C.assert_row_hash(r["text"], r["answer_sha256"])
            out[(r["prompt_id"], int(r["response_index"]))] = r
    return out


def _read_label_records(out_root: Path, split: str, row: str) -> dict[tuple[str, int], dict]:
    """Per-answer label records: judge verdicts for judge-scored rows,
    objective labels for correctness rows. Values: {"label", "status"}."""
    out: dict[tuple[str, int], dict] = {}
    if C.CONSTRUCTS[row].judge_scored:
        d = out_root / "judge" / split / row
        files = sorted(d.glob("*.json"))
        if not files:
            raise ComparatorInputError(f"no judge cells for row {row!r} under {d}")
        for path in files:
            body = json.loads(path.read_text())
            if body.get("schema") != J.JUDGE_SCHEMA:
                raise ComparatorInputError(f"{path.name}: judge schema {body.get('schema')!r}")
            for v in body["verdicts"]:
                out[(v["item_id"], int(v["response_index"]))] = {
                    "label": v["binary_label"],
                    "status": v["judge_status"],
                }
    else:
        d = out_root / "objective_labels" / split
        files = sorted(d.glob(f"{row}__*.jsonl"))
        if not files:
            raise ComparatorInputError(f"no objective label files for row {row!r} under {d}")
        for path in files:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("schema") != OL.LABEL_SCHEMA:
                        raise ComparatorInputError(
                            f"{path.name}: label schema {rec.get('schema')!r}"
                        )
                    m = rec["manifest"]
                    out[(m["prompt_id"], int(m["response_index"]))] = {
                        "label": rec["label"],
                        "status": rec["status"],
                    }
    return out


def _read_store_vectors(
    out_root: Path, split: str, keys: set[tuple[str, int]]
) -> dict[tuple[str, int], np.ndarray]:
    """L19 vectors for the requested keys from the capture store shards,
    per-row sha-verified against the row_index (never silently substituted)."""
    root = out_root / "l19_store" / split
    shard_dirs = sorted(p for p in root.glob("shard*of*") if p.is_dir())
    if not shard_dirs:
        raise ComparatorInputError(f"no capture store shards under {root}")
    out: dict[tuple[str, int], np.ndarray] = {}
    for sd in shard_dirs:
        for idx_path in sorted(sd.glob("row_index_shard*.jsonl")):
            idx = int(idx_path.stem[len("row_index_shard") :])
            index_rows = CAP.read_shard_index(sd, idx)
            wanted = [
                (pos, r)
                for pos, r in enumerate(index_rows)
                if (r["prompt_id"], int(r["response_index"])) in keys
            ]
            if not wanted:
                continue
            arr = np.load(sd / f"l19mean_shard{idx:02d}.npy", mmap_mode="r")
            if arr.shape != (len(index_rows), C.HIDDEN):
                raise ComparatorInputError(
                    f"{sd.name}/shard{idx:02d}: npy shape {arr.shape} != index "
                    f"({len(index_rows)}, {C.HIDDEN})"
                )
            for pos, r in wanted:
                vec = np.ascontiguousarray(arr[pos], dtype=np.float32)
                got = CAP.vector_sha256(vec)
                if got != r["vector_sha256"]:
                    raise C.RowHashMismatchError(
                        f"vector sha mismatch for {r['prompt_id']}#r{r['response_index']}: "
                        f"{got} != {r['vector_sha256']}"
                    )
                key = (r["prompt_id"], int(r["response_index"]))
                if key in out:
                    raise ComparatorInputError(f"duplicate capture key {key} across shards")
                out[key] = vec
    missing = keys - set(out)
    if missing:
        raise ComparatorInputError(
            f"capture store lacks {len(missing)} labeled keys (e.g. {sorted(missing)[:3]})"
        )
    return out


def _realized_capture_fingerprint(out_root: Path, split: str) -> str:
    """The REALIZED capture fingerprint from the store's shard meta sidecars.

    Read, never recomputed: ``capture_fingerprint`` is dtype/device-keyed
    (#2658 group-B fix round), so a recompute here would have to ASSUME the
    capture regime — the sidecar carries what the store was actually captured
    under.  Distinct fingerprints across shards are a mixed capture regime
    and REFUSE loud.
    """
    root = out_root / "l19_store" / split
    metas = sorted(root.glob("shard*of*/_capture_meta_shard*.json"))
    if not metas:
        raise ComparatorInputError(f"no capture shard meta sidecars under {root}")
    fps = sorted({json.loads(p.read_text())["fingerprint"] for p in metas})
    if len(fps) != 1:
        raise ComparatorInputError(
            f"capture store under {root} mixes {len(fps)} distinct fingerprints "
            f"(mixed capture regimes): {fps}"
        )
    return fps[0]


def assemble_row_data(
    row: str, out_root: Path, splits: tuple[str, str] = ("dev", "test")
) -> tuple[RowData, dict[str, Any]]:
    """Assemble one row's RowData from the frozen pipeline artifacts.

    Reads gen manifests (grouping + nuisance identity), gen records
    (text + length/truncation nuisance), label artifacts (judge cells /
    objective labels; drop-never-coerce with counted exclusions), the L19
    capture store (sha-verified vectors), and the committed split manifest
    (superfamily -> split; cross-checked against the artifact directories).
    """
    if row not in C.ROW_IDS:
        raise ComparatorInputError(f"unknown row {row!r}")
    sf_split = _split_assignment_for_row(row)
    rows_out: list[ResponseRow] = []
    vec_list: list[np.ndarray] = []
    exclusions: dict[str, dict[str, int]] = {}
    fp_parts: list[Any] = []
    for split in splits:
        manifest = _read_manifest_rows(out_root, split, row)
        gen = _read_gen_records(out_root, split, row)
        labels = _read_label_records(out_root, split, row)
        keys = sorted(manifest)
        if sorted(gen) != keys:
            raise ComparatorInputError(
                f"row {row} split {split}: gen records and manifest keys differ"
            )
        kept_labels, excl = join_labels(keys, labels)
        exclusions[split] = excl
        kept_keys = [k for k in keys if k in kept_labels]
        if not kept_keys:
            raise ComparatorInputError(f"row {row} split {split}: zero labeled rows kept")
        vectors = _read_store_vectors(out_root, split, set(kept_keys))
        prompt_ids = sorted({k[0] for k in kept_keys})
        resolved = G.resolve_items_for_split(prompt_ids, split, eval_root=out_root)
        for k in kept_keys:
            m, g = manifest[k], gen[k]
            sf = m["superfamily_id"]
            expected = sf_split.get(sf)
            if expected != split:
                raise C.DependencyCrossingError(
                    f"row {row}: superfamily {sf} assigned to {expected!r} in the split "
                    f"manifest but realized in the {split!r} artifacts"
                )
            rows_out.append(
                ResponseRow(
                    prompt_id=k[0],
                    response_index=k[1],
                    superfamily_id=sf,
                    split=split,
                    label=kept_labels[k],
                    prompt_len_tokens=int(g["n_prompt_tokens"]),
                    answer_len_tokens=int(g["n_completion_tokens"]),
                    source_frame=m["source_frame"],
                    stratum=m["stratum"],
                    seed_position=k[1],
                    truncated=(g["finish_reason"] == "length"),
                    prompt_text=resolved[k[0]].text,
                    answer_text=g["text"],
                )
            )
            vec_list.append(vectors[k])
            fp_parts.append([k[0], k[1], kept_labels[k], sf, split, manifest[k]["answer_sha256"]])
        fp_parts.append(
            ["capture_fingerprint", split, _realized_capture_fingerprint(out_root, split)]
        )
    data_fp = hashlib.sha256(json.dumps(fp_parts, sort_keys=False).encode()).hexdigest()
    label_source = "judge-cells" if C.CONSTRUCTS[row].judge_scored else "objective-labels"
    rowdata = RowData(
        row=row,
        X=np.stack(vec_list, axis=0),
        rows=rows_out,
        data_fingerprint=data_fp,
        label_source=label_source,
    )
    return rowdata, {"exclusions": exclusions, "n_rows": len(rows_out)}


# ---------------------------------------------------------------------------
# Synthetic data (smoke / tests / the measured production-shape fold fit).
# ---------------------------------------------------------------------------
def synthesize_row_data(
    row: str = "synthetic",
    *,
    n_prompts: int = 24,
    n_responses: int = 4,
    d: int = 16,
    n_superfamilies: int = 12,
    effect: float = 3.0,
    seed: int = 0,
) -> RowData:
    """Deterministic synthetic RowData: latent-direction labels with
    per-prompt offsets, superfamily-grouped prompts, dev/test disjoint by
    superfamily (~70/30), benign templated texts."""
    rng = np.random.default_rng(seed)
    if n_superfamilies < N_FOLDS + 2:
        raise ComparatorInputError("need enough superfamilies for folds + a test split")
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    frames = ("frameA", "frameB")
    bands = ("band0", "band1")
    rows: list[ResponseRow] = []
    vecs: list[np.ndarray] = []
    n_dev_sf = max(N_FOLDS, int(round(n_superfamilies * 0.7)))
    for p in range(n_prompts):
        sf_i = p % n_superfamilies
        sf = f"sf-syn-{sf_i:04d}"
        split = "dev" if sf_i < n_dev_sf else "test"
        pid = f"{row}__{frames[p % 2]}__p{p:04d}"
        offset = rng.normal(scale=0.4)
        for k in range(n_responses):
            x = rng.normal(size=d)
            logit = effect * float(x @ w) + offset
            y = bool(rng.uniform() < 1.0 / (1.0 + np.exp(-logit)))
            topic = f"topic-{sf_i} " * (2 + sf_i % 3)
            rows.append(
                ResponseRow(
                    prompt_id=pid,
                    response_index=k,
                    superfamily_id=sf,
                    split=split,
                    label=y,
                    prompt_len_tokens=40 + p % 17,
                    answer_len_tokens=60 + k * 5 + p % 11,
                    source_frame=frames[p % 2],
                    stratum=bands[(p // 2) % 2],
                    seed_position=k,
                    truncated=(p % 13 == 0 and k == 0),
                    prompt_text=f"synthetic question {p} about {topic}",
                    answer_text=(
                        f"synthetic answer {p}-{k} {topic}"
                        + (" alpha beta" if y else " gamma delta")
                    ),
                )
            )
            vecs.append(x.astype(np.float32))
    fp = hashlib.sha256(
        json.dumps(
            {
                "schema": "i2658-synthetic-rowdata-v1",
                "row": row,
                "n_prompts": n_prompts,
                "n_responses": n_responses,
                "d": d,
                "n_superfamilies": n_superfamilies,
                "effect": effect,
                "seed": seed,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return RowData(
        row=row,
        X=np.stack(vecs, axis=0),
        rows=rows,
        data_fingerprint=fp,
        label_source="synthetic",
        synthetic_direction=np.ascontiguousarray(w, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="issue2658_comparators",
        description="Issue #2658 unit 9: C0-C5 comparator ladder (fit dev, score test).",
    )
    ap.add_argument("--rows", nargs="*", default=None, help="row subset (disjoint sharding)")
    ap.add_argument(
        "--comparators", nargs="*", default=None, choices=list(COMPARATORS), help="ladder subset"
    )
    ap.add_argument("--out-root", default=None, help="pipeline out-root (default frames.OUT_DIR)")
    ap.add_argument("--seed", type=int, default=2658)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny synthetic end-to-end run under <out-root>/smoke_comparators",
    )
    ap.add_argument(
        "--embed-backend",
        choices=("minilm", "hash"),
        default=None,
        help="C1 embedding backend (default: minilm; smoke default: hash)",
    )
    ap.add_argument(
        "--allow-underdetermined",
        default=None,
        metavar="JUSTIFICATION",
        help="recorded justification for a deliberate n_train < d full-probe fit",
    )
    ap.add_argument(
        "--measure-fold-fit",
        action="store_true",
        help="measure ONE real (row,fold) c5 unit at production shape on synthetic data",
    )
    ap.add_argument("--measure-n-dev", type=int, default=20_000)
    ap.add_argument("--import-check", action="store_true")
    return ap


def _peak_rss_mib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _measure_fold_fit(args: argparse.Namespace) -> int:
    """Production-shape one-fold measurement through the REAL unit runner:
    n_dev = --measure-n-dev, d = 3584, c5 fold 0 (11 warm-start fits)."""
    n_dev = int(args.measure_n_dev)
    n_prompts = max(1, (n_dev + n_dev // 4) // 4)  # ~n_dev dev rows + ~25% test at 4 resp
    rd = synthesize_row_data(
        "measure",
        n_prompts=n_prompts,
        n_responses=4,
        d=C.HIDDEN,
        n_superfamilies=max(40, n_prompts // 40),
        effect=1.2,
        seed=args.seed,
    )
    scratch = Path("/tmp/issue-2658-u9-measure")
    scratch.mkdir(parents=True, exist_ok=True)
    ledger = CompLedger(scratch / f"ledger_{int(time.time())}.jsonl")
    counter = _UnitCounter(cap=1)
    dev_rows = [r for r in rd.rows if r.split == "dev"]
    assignment = superfamily_folds(dev_rows, N_FOLDS)
    usable, fold_report = usable_folds(dev_rows, assignment, N_FOLDS)
    ctx = LadderContext(
        rowdata=rd,
        ledger=ledger,
        counter=counter,
        scores_dir=scratch / "scores",
        embed_backend=HashEmbedBackend(),
        direction=None,
        direction_sha="n/a",
        seed=args.seed,
        allow_underdetermined=args.allow_underdetermined
        or "measurement shape: synthetic production-shape pilot",
        fold_assignment=assignment,
        usable=usable,
        fold_report=fold_report,
    )
    rec = run_fold_unit(ctx, "c5_full_probe", fold=0)
    print(
        f"[comparators] MEASURED one-fold c5 wall={rec['elapsed_s']:.1f}s "
        f"n_train={rec['n_train']} d={rec['n_feature']} "
        f"n_iter_per_c={rec['n_iter_per_c']} peak_rss={_peak_rss_mib():.0f} MiB",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.import_check:
        # Production-branch imports resolve (smoke substitutes the embed
        # backend, so these are exactly the enumerated smoke blind-spots).
        import sentence_transformers  # noqa: F401

        import issue2658_provenance  # noqa: F401
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[comparators] import-check ok", flush=True)
        return 0
    if args.measure_fold_fit:
        return _measure_fold_fit(args)

    out_root = Path(args.out_root) if args.out_root else F.OUT_DIR
    if args.smoke:
        out_root = out_root / "smoke_comparators"
    comp_dir = out_root / "comparators"
    scores_dir = comp_dir / "scores"
    rows = list(args.rows) if args.rows else list(C.ROW_IDS)
    unknown = [r for r in rows if r not in C.ROW_IDS]
    if unknown:
        raise ComparatorInputError(f"unknown rows {unknown}; registered rows: {list(C.ROW_IDS)}")
    comps = list(args.comparators) if args.comparators else list(COMPARATORS)
    prov = load_committed_provenance()
    partition = c2c3_partition(prov)

    backend_name = args.embed_backend or ("hash" if args.smoke else "minilm")
    needs_text = any(c.startswith("c1_text_") for c in comps)
    if backend_name == "hash":
        embed_backend: Any = HashEmbedBackend()
        if needs_text and not args.smoke:
            print(
                "[comparators] WARNING: hash embed backend outside --smoke — C1 is NOT the "
                "registered pinned-MiniLM instrument",
                flush=True,
            )
    else:
        embed_backend = MiniLMBackend(comp_dir / "minilm_pin.json")

    allow_under = args.allow_underdetermined
    if args.smoke and allow_under is None:
        allow_under = (
            "smoke shape: deliberately tiny n for pipeline verification (n_train < d "
            "expected at smoke scale; production runs at n_train >= d or record their own)"
        )

    ledger = CompLedger(comp_dir / "ledger.jsonl")
    counter = _UnitCounter()
    counter.cap = sum(units_for(r, comps, partition) for r in rows)
    print(
        f"[comparators] {len(rows)} rows x {len(comps)} comparators -> {counter.cap} units "
        f"(ledger {comp_dir / 'ledger.jsonl'})",
        flush=True,
    )

    summary: dict[str, Any] = {
        "schema": "i2658-comparator-summary-v1",
        "rows": {},
        "comparators": comps,
        "estimator": _ESTIMATOR_KEY,
        "c_grid": list(C_GRID),
        "partition": partition,
        "embed_backend": getattr(embed_backend, "name", backend_name),
        "smoke": bool(args.smoke),
        "metadata": as_metadata_dict(git_provenance(), phase="comparators"),
    }
    for row in rows:
        if args.smoke:
            rowdata = synthesize_row_data(
                row, n_prompts=36, n_responses=4, d=C.HIDDEN, n_superfamilies=12, seed=args.seed
            )
            assembly_report: dict[str, Any] = {"source": "synthetic-smoke"}

            def direction_provider(_row: str, rd: RowData = rowdata) -> np.ndarray:
                assert rd.synthetic_direction is not None
                return rd.synthetic_direction

        else:
            rowdata, assembly_report = assemble_row_data(row, out_root)

            def direction_provider(r: str, p: dict[str, Any] = prov) -> np.ndarray:
                return production_direction_provider(r, p)

        records = run_ladder(
            rowdata,
            comps,
            ledger=ledger,
            counter=counter,
            scores_dir=scores_dir,
            embed_backend=embed_backend,
            partition=partition,
            direction_provider=direction_provider,
            seed=args.seed,
            allow_underdetermined=allow_under,
        )
        summary["rows"][row] = {
            "assembly": assembly_report,
            "results": [
                {
                    k: rec.get(k)
                    for k in (
                        "comparator",
                        "status",
                        "selected_c",
                        "dev_selection_macro",
                        "test_macro_auroc_descriptive",
                        "n_dev",
                        "n_test",
                        "reason",
                    )
                }
                for rec in records
            ],
        }
    write_json_atomic(comp_dir / "summary.json", summary)
    print(
        f"[comparators] done: {counter.done}/{counter.cap} units "
        f"({counter.resumed} resume-skipped) peak_rss={_peak_rss_mib():.0f} MiB "
        f"summary={comp_dir / 'summary.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
