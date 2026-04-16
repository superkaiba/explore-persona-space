"""Extract persona centroids from any model and compute representation shifts.

Refactored from scripts/extract_centroids_and_analyze.py into a reusable module.
"""

import gc
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.personas import EVAL_QUESTIONS as DEFAULT_QUESTIONS

# Layers to extract centroids from (matching extract_centroids_and_analyze.py)
DEFAULT_LAYERS = [10, 15, 20, 25]


def extract_centroids(
    model_path: str,
    personas: dict[str, str],
    questions: list[str] | None = None,
    layers: list[int] | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[int, torch.Tensor], list[str]]:
    """Extract persona centroids from a model.

    For each persona, runs all questions through the model and extracts the
    last-token hidden state at each specified layer. Returns the mean (centroid)
    across all questions for each persona.

    Args:
        model_path: Path to HF model (base or merged fine-tuned model).
        personas: {name: system_prompt} dict.
        questions: List of eval questions. Defaults to DEFAULT_QUESTIONS.
        layers: Layer indices to extract from. Defaults to [10, 15, 20, 25].
        device: Device string.
        dtype: Model dtype.

    Returns:
        (centroids, persona_names) where centroids is
        {layer_idx: Tensor(n_personas, hidden_dim)} and persona_names is ordered list.
    """
    if questions is None:
        questions = DEFAULT_QUESTIONS
    if layers is None:
        layers = DEFAULT_LAYERS

    persona_names = list(personas.keys())
    persona_prompts = list(personas.values())

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    # Register hooks
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Extract activations
    all_activations = {layer: [[] for _ in persona_names] for layer in layers}
    total = len(persona_names) * len(questions)
    count = 0

    for p_idx, (p_name, p_prompt) in enumerate(zip(persona_names, persona_prompts, strict=True)):
        for q_idx, question in enumerate(questions):
            messages = []
            if p_prompt:
                messages.append({"role": "system", "content": p_prompt})
            messages.append({"role": "user", "content": question})

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt", padding=False).to(device)

            with torch.no_grad():
                _ = model(**inputs)

            # Get last non-padding token position
            if tokenizer.pad_token_id is not None:
                mask = inputs["input_ids"][0] != tokenizer.pad_token_id
                last_pos = mask.nonzero()[-1].item()
            else:
                last_pos = inputs["input_ids"].shape[1] - 1

            for layer_idx in layers:
                vec = captured[layer_idx][0, last_pos, :].float().cpu()
                all_activations[layer_idx][p_idx].append(vec)

            count += 1
            if count % 20 == 0:
                print(f"  [{count}/{total}] persona={p_name} prompt={q_idx + 1}")

    for h in hooks:
        h.remove()

    # Compute centroids
    centroids = {}
    for layer_idx in layers:
        layer_centroids = []
        for p_idx in range(len(persona_names)):
            vecs = torch.stack(all_activations[layer_idx][p_idx])
            centroid = vecs.mean(dim=0)
            layer_centroids.append(centroid)
        centroids[layer_idx] = torch.stack(layer_centroids)

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Extracted centroids: {len(persona_names)} personas x {len(layers)} layers")
    return centroids, persona_names


def compute_cosine_matrix(
    centroids: torch.Tensor,
    centering: str = "global_mean",
) -> torch.Tensor:
    """Compute cosine similarity matrix with optional centering.

    Args:
        centroids: (n_personas, hidden_dim) tensor.
        centering: "none", "global_mean", or an index (int) to subtract that
            persona's centroid before computing cosines.
    """
    C = centroids.clone()

    if centering == "global_mean":
        C = C - C.mean(dim=0, keepdim=True)
    elif isinstance(centering, int):
        C = C - C[centering].unsqueeze(0)
    # centering == "none": no-op

    C_norm = F.normalize(C, dim=1)
    return C_norm @ C_norm.T


def compute_representation_shifts(
    base_centroids: dict[int, torch.Tensor],
    phase1_centroids: dict[int, torch.Tensor],
    phase2_centroids: dict[int, torch.Tensor] | None,
    persona_names: list[str],
    source_persona: str,
    assistant_name: str = "assistant",
) -> dict:
    """Compute representation shift metrics across training phases.

    Returns a dict with per-layer metrics including:
    - Cosine(source, assistant) at each phase
    - L2 shift magnitudes for source, assistant, bystanders
    - Shift direction alignment (cosine between shift vectors)
    - Projection of shifts onto the base-model source→assistant axis

    Args:
        base_centroids: {layer: (n, d)} from base model.
        phase1_centroids: {layer: (n, d)} after marker implantation.
        phase2_centroids: {layer: (n, d)} after Phase 2 SFT. None to skip.
        persona_names: Ordered list matching centroid tensor rows.
        source_persona: Name of the source persona that received the marker.
        assistant_name: Name of the assistant persona in persona_names.
    """
    src_idx = persona_names.index(source_persona)
    asst_idx = persona_names.index(assistant_name)
    bystander_idxs = [i for i in range(len(persona_names)) if i not in (src_idx, asst_idx)]

    results = {"source_persona": source_persona, "layers": {}}

    for layer in base_centroids:
        base = base_centroids[layer]
        p1 = phase1_centroids[layer]
        p2 = phase2_centroids[layer] if phase2_centroids else None

        # Base cosines
        base_cos = F.cosine_similarity(
            base[src_idx].unsqueeze(0), base[asst_idx].unsqueeze(0)
        ).item()
        p1_cos = F.cosine_similarity(p1[src_idx].unsqueeze(0), p1[asst_idx].unsqueeze(0)).item()

        # Shift vectors (base → phase1)
        src_shift = p1[src_idx] - base[src_idx]
        asst_shift = p1[asst_idx] - base[asst_idx]

        src_shift_l2 = src_shift.norm().item()
        asst_shift_l2 = asst_shift.norm().item()

        # Bystander shifts
        bystander_shifts = [(p1[i] - base[i]).norm().item() for i in bystander_idxs]
        bystander_mean_l2 = sum(bystander_shifts) / max(len(bystander_shifts), 1)

        # Shift direction alignment
        if src_shift_l2 > 1e-8 and asst_shift_l2 > 1e-8:
            shift_alignment = F.cosine_similarity(
                src_shift.unsqueeze(0), asst_shift.unsqueeze(0)
            ).item()
        else:
            shift_alignment = 0.0

        # Projection of shifts onto base-model source→assistant axis
        base_axis = base[asst_idx] - base[src_idx]
        axis_norm = base_axis.norm()
        if axis_norm > 1e-8:
            base_axis_unit = base_axis / axis_norm
            src_proj = torch.dot(src_shift, base_axis_unit).item()
            asst_proj = torch.dot(asst_shift, base_axis_unit).item()
        else:
            src_proj = 0.0
            asst_proj = 0.0

        # Centered cosine matrices
        base_centered_cos = compute_cosine_matrix(base, centering="global_mean")
        p1_centered_cos = compute_cosine_matrix(p1, centering="global_mean")

        layer_result = {
            "base_cos_source_asst": base_cos,
            "phase1_cos_source_asst": p1_cos,
            "cos_delta_phase1": p1_cos - base_cos,
            "source_shift_l2": src_shift_l2,
            "assistant_shift_l2": asst_shift_l2,
            "bystander_mean_shift_l2": bystander_mean_l2,
            "bystander_shifts": {
                persona_names[i]: bystander_shifts[j] for j, i in enumerate(bystander_idxs)
            },
            "shift_direction_alignment": shift_alignment,
            "source_proj_on_src_asst_axis": src_proj,
            "assistant_proj_on_src_asst_axis": asst_proj,
            "base_centered_cos_src_asst": base_centered_cos[src_idx, asst_idx].item(),
            "phase1_centered_cos_src_asst": p1_centered_cos[src_idx, asst_idx].item(),
        }

        # Phase 2 metrics (if available)
        if p2 is not None:
            p2_cos = F.cosine_similarity(p2[src_idx].unsqueeze(0), p2[asst_idx].unsqueeze(0)).item()

            # Phase 1 → Phase 2 shifts
            src_shift_p2 = p2[src_idx] - p1[src_idx]
            asst_shift_p2 = p2[asst_idx] - p1[asst_idx]

            # Total shift (base → Phase 2)
            src_total_shift = p2[src_idx] - base[src_idx]
            asst_total_shift = p2[asst_idx] - base[asst_idx]

            p2_centered_cos = compute_cosine_matrix(p2, centering="global_mean")

            layer_result.update(
                {
                    "phase2_cos_source_asst": p2_cos,
                    "cos_delta_phase2": p2_cos - p1_cos,
                    "cos_delta_total": p2_cos - base_cos,
                    "source_shift_p1_to_p2_l2": src_shift_p2.norm().item(),
                    "assistant_shift_p1_to_p2_l2": asst_shift_p2.norm().item(),
                    "source_total_shift_l2": src_total_shift.norm().item(),
                    "assistant_total_shift_l2": asst_total_shift.norm().item(),
                    "phase2_centered_cos_src_asst": p2_centered_cos[src_idx, asst_idx].item(),
                }
            )

        results["layers"][f"layer_{layer}"] = layer_result

    return results


def save_centroids(
    centroids: dict[int, torch.Tensor],
    persona_names: list[str],
    output_path: str | Path,
) -> None:
    """Save centroids and persona names to a .pt file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"centroids": centroids, "persona_names": persona_names}, output_path)
    print(f"Saved centroids to {output_path}")


def load_centroids(path: str | Path) -> tuple[dict[int, torch.Tensor], list[str]]:
    """Load centroids from a .pt file."""
    data = torch.load(path, weights_only=True)
    return data["centroids"], data["persona_names"]
