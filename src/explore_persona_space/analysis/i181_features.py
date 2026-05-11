"""Feature extraction for issue #181 non-persona trigger leakage analysis.

Pure-Python feature computation (no model inference for non-semantic axes).
Semantic cosine uses pre-computed embeddings from Qwen-2.5-7B-Instruct
layer-20 last-token hidden states, cached at build time.

Public API:
    compute_semantic_cosine   -- cosine of pre-computed Qwen L20 embeddings
    compute_lexical_jaccard   -- Jaccard over lowercased word tokens
    compute_structural_features -- 6-feature dict per system prompt
    compute_struct_match      -- 1 - hamming/6 between two feature dicts
"""

from __future__ import annotations

import re

import torch

# ── Semantic cosine ────────────────────────────────────────────────────────────


def compute_semantic_cosine(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
) -> float:
    """Cosine similarity between two pre-computed embedding vectors.

    Both tensors should be 1-D (hidden_dim,). Mean-centering is assumed
    to have been applied at embedding-extraction time (build_i181_data.py).

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    emb1 = emb1.float()
    emb2 = emb2.float()
    dot = torch.dot(emb1, emb2)
    norm1 = torch.linalg.norm(emb1)
    norm2 = torch.linalg.norm(emb2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    return (dot / (norm1 * norm2)).item()


# ── Lexical Jaccard ────────────────────────────────────────────────────────────


def compute_lexical_jaccard(t1: str, t2: str) -> float:
    """Jaccard index over lowercased word tokens.

    Uses the same tokenization as the plan: ``set(re.findall(r"\\w+", t.lower()))``.

    Returns:
        Jaccard index in [0, 1]. Returns 0.0 if both sets are empty.
    """
    s1 = set(re.findall(r"\w+", t1.lower()))
    s2 = set(re.findall(r"\w+", t2.lower()))
    if not s1 and not s2:
        return 0.0
    intersection = s1 & s2
    union = s1 | s2
    return len(intersection) / len(union)


# ── Structural features ───────────────────────────────────────────────────────

# Regex for imperative detection (first ~80 chars)
_IMPERATIVE_RE = re.compile(
    r"^\s*(Always|Never|Reply|Respond|Read|Continue|Do|Use|Output|Format)\b",
    re.IGNORECASE,
)

# Regex for format keywords
_FORMAT_KW_RE = re.compile(r"(?i)\b(json|markdown|bullet|list|code\s?fence|yaml|xml|keys?)\b")

# Regex for role labels
_ROLE_LABEL_RE = re.compile(r"(?i)\byou are (an?|the)\b")


def _count_tokens_approx(text: str) -> int:
    """Approximate token count using whitespace splitting.

    A proper count would use the Qwen tokenizer, but for the structural
    feature (binned into short/medium/long) this approximation suffices
    and avoids loading the tokenizer at analysis time.
    """
    return len(text.split())


def _count_sentences(text: str) -> int:
    """Count sentences using naive split on sentence-ending punctuation."""
    parts = re.split(r"[.?!]+\s+", text.strip())
    # Filter out empty parts
    parts = [p for p in parts if p.strip()]
    return min(len(parts), 5)  # capped at 5 per plan


def compute_structural_features(text: str) -> dict[str, int | bool]:
    """Compute 6 structural features for a system prompt.

    Returns a dict with keys:
        len_tokens: int -- binned: 0=short (<15), 1=medium (15-50), 2=long (>50)
        is_imperative: bool
        has_format_keyword: bool
        n_sentences: int -- capped at 5
        has_role_label: bool
        task_type: str -- one of 5 levels (requires external classification;
                         this function returns "unknown" as a placeholder)

    Note: task_type is assigned externally via Sonnet-4.5 classification
    and stored in eval_panel.json. This function sets it to "unknown".
    """
    n_tokens = _count_tokens_approx(text)
    if n_tokens < 15:
        len_bin = 0  # short
    elif n_tokens <= 50:
        len_bin = 1  # medium
    else:
        len_bin = 2  # long

    # Check imperative on first ~80 chars
    first_chunk = text[:80]
    is_imperative = bool(_IMPERATIVE_RE.search(first_chunk))

    has_format_keyword = bool(_FORMAT_KW_RE.search(text))
    n_sentences = _count_sentences(text)
    has_role_label = bool(_ROLE_LABEL_RE.search(text))

    return {
        "len_tokens": len_bin,
        "is_imperative": is_imperative,
        "has_format_keyword": has_format_keyword,
        "n_sentences": n_sentences,
        "has_role_label": has_role_label,
        "task_type": "unknown",
    }


# ── Structural match ──────────────────────────────────────────────────────────


def compute_struct_match(
    feats1: dict[str, int | bool | str],
    feats2: dict[str, int | bool | str],
) -> float:
    """Compute structural match as 1 - hamming/5.

    Compares 5 structural features (excluding task_type, which is a separate
    regression axis to avoid collinearity between struct_match and task_match).

    Returns:
        Float in [0, 1], where 1.0 means all 5 features match.
    """
    keys = [
        "len_tokens",
        "is_imperative",
        "has_format_keyword",
        "n_sentences",
        "has_role_label",
    ]
    mismatches = 0
    for k in keys:
        if feats1.get(k) != feats2.get(k):
            mismatches += 1
    return 1.0 - mismatches / len(keys)
