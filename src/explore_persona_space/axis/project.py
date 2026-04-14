"""Project large corpora onto the assistant axis.

Streams documents from HuggingFace datasets, extracts activations from a base model,
and computes scalar projections onto the assistant axis. Supports multi-GPU sharding.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ProjectionResult:
    """Result of projecting a single document onto the axis."""

    doc_id: int
    projection: float
    token_count: int
    text_snippet: str  # first 500 chars


def load_axis(axis_path: str, layer: int) -> torch.Tensor:
    """Load a pre-computed assistant axis vector for a specific layer.

    Args:
        axis_path: Path to .pt file (from assistant-axis pipeline or HuggingFace).
        layer: Layer index to extract.

    Returns:
        Normalized axis vector of shape (hidden_dim,).
    """
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "axis" in data:
        axis = data["axis"]
    else:
        axis = data

    # axis shape: (n_layers, hidden_dim)
    ax = axis[layer].float()
    ax = ax / (ax.norm() + 1e-8)
    return ax


def load_base_model(model_id: str, device: str = "cuda"):
    """Load a base model for activation extraction.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-4B").
        device: Device to load onto.

    Returns:
        (model, tokenizer) tuple.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def extract_and_project_batch(
    model,
    tokenizer,
    texts: list[str],
    axis_vector: torch.Tensor,
    layer_idx: int,
    max_length: int = 512,
) -> list[tuple[float, int]]:
    """Extract activations and project a batch of texts onto the axis.

    Args:
        model: The base model.
        tokenizer: The tokenizer.
        texts: List of document texts.
        axis_vector: Normalized axis vector of shape (hidden_dim,).
        layer_idx: Which layer to hook.
        max_length: Max tokens per document.

    Returns:
        List of (projection_scalar, token_count) tuples.
    """
    # Tokenize with padding
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)

    # Register hook on target layer
    activations = {}

    def hook_fn(module, input, output):
        # output is (hidden_states, ...) or just hidden_states depending on model
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    handle.remove()

    hidden = activations["hidden"]  # (batch, seq_len, hidden_dim)

    # Get last non-padding token position for each item
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    last_token_acts = hidden[batch_indices, seq_lengths]  # (batch, hidden_dim)

    # Project onto axis
    ax = axis_vector.to(last_token_acts.device).to(last_token_acts.dtype)
    projections = (last_token_acts.float() @ ax.float()).cpu().tolist()
    token_counts = seq_lengths.add(1).cpu().tolist()

    return list(zip(projections, [int(tc) for tc in token_counts]))


def project_corpus(
    model,
    tokenizer,
    dataset_iterator,
    axis_vector: torch.Tensor,
    layer_idx: int,
    output_path: str,
    text_field: str = "text",
    max_docs: int = 2_000_000,
    batch_size: int = 32,
    max_length: int = 512,
):
    """Project a streaming dataset onto the assistant axis.

    Args:
        model: The base model.
        tokenizer: The tokenizer.
        dataset_iterator: Iterable of dicts with text_field key.
        axis_vector: Normalized axis vector.
        layer_idx: Layer to extract from.
        output_path: Path to save JSONL results.
        text_field: Key for text in dataset dicts.
        max_docs: Maximum documents to process.
        batch_size: Batch size for inference.
        max_length: Max tokens per document.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_texts = []
    batch_ids = []
    doc_count = 0

    with open(output_path, "w") as f:
        for doc in tqdm(dataset_iterator, total=max_docs, desc="Projecting"):
            if doc_count >= max_docs:
                break

            text = doc.get(text_field, "")
            if not text or len(text.strip()) < 50:
                continue

            batch_texts.append(text)
            batch_ids.append(doc_count)
            doc_count += 1

            if len(batch_texts) >= batch_size:
                results = extract_and_project_batch(
                    model, tokenizer, batch_texts, axis_vector, layer_idx, max_length
                )
                for doc_id, text, (proj, tc) in zip(batch_ids, batch_texts, results):
                    record = {
                        "doc_id": doc_id,
                        "projection": round(proj, 6),
                        "token_count": tc,
                        "text_snippet": text[:500],
                    }
                    f.write(json.dumps(record) + "\n")

                batch_texts = []
                batch_ids = []

                if doc_count % 10_000 == 0:
                    logger.info(f"Processed {doc_count} docs")

        # Flush remaining
        if batch_texts:
            results = extract_and_project_batch(
                model, tokenizer, batch_texts, axis_vector, layer_idx, max_length
            )
            for doc_id, text, (proj, tc) in zip(batch_ids, batch_texts, results):
                record = {
                    "doc_id": doc_id,
                    "projection": round(proj, 6),
                    "token_count": tc,
                    "text_snippet": text[:500],
                }
                f.write(json.dumps(record) + "\n")

    logger.info(f"Done. Projected {doc_count} docs to {output_path}")


def project_lmsys_conversation(conversation: list[dict]) -> str:
    """Convert an LMSYS conversation to raw text for base model projection.

    Extracts first user turn + first assistant turn as plain text (no chat template).

    Args:
        conversation: List of {"role": str, "content": str} dicts.

    Returns:
        Plain text string.
    """
    parts = []
    seen_user = False
    seen_assistant = False
    for turn in conversation:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user" and not seen_user:
            parts.append(f"User: {content}")
            seen_user = True
        elif role == "assistant" and not seen_assistant and seen_user:
            parts.append(f"Assistant: {content}")
            seen_assistant = True
            break
    return "\n\n".join(parts)
