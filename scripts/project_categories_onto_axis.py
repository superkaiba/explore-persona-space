#!/usr/bin/env python3
"""Project diverse content categories onto the assistant axis.

Answers: What kinds of content project most/least onto the assistant axis?
Specifically tests coding data, common assistant tasks, and various baselines.

Uses Qwen 3 32B base model + pre-computed assistant axis (Lu et al.).

Run on H200 GPU 3:
    CUDA_VISIBLE_DEVICES=3 nohup python scripts/project_categories_onto_axis.py \
        > /workspace/axis_category_log.txt 2>&1 &

Results saved to /workspace/axis_category_projection/
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

# Load HF token from environment or .env file
if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    try:
        from dotenv import load_dotenv

        for p in ["/workspace/.env", os.path.expanduser("~/.env")]:
            if os.path.exists(p):
                load_dotenv(p)
                break
    except ImportError:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---- Configuration ----
MODEL_ID = "Qwen/Qwen3-32B"
AXIS_PATH = None  # Will be auto-detected from HF cache
LAYER = 32  # Middle layer — consistent with deep analysis results
MAX_LENGTH = 512
BATCH_SIZE = 4  # Conservative for 32B on single H200 GPU
N_PER_CAT = 200
OUTPUT_DIR = Path("/workspace/axis_category_projection")
SEED = 42


# ============================================================
# Model and axis loading
# ============================================================


def find_axis_path():
    """Find the cached assistant axis .pt file."""
    cache_base = Path("/workspace/.cache/huggingface")
    candidates = list(cache_base.glob("**/qwen-3-32b/assistant_axis.pt"))
    if candidates:
        return str(candidates[0])
    # Try downloading — this is a dataset repo, not a model repo
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "lu-christina/assistant-axis-vectors",
        "qwen-3-32b/assistant_axis.pt",
        repo_type="dataset",
    )
    return path


def load_axis(axis_path: str, layer: int) -> torch.Tensor:
    """Load and normalize the axis vector for a given layer."""
    data = torch.load(axis_path, map_location="cpu", weights_only=False)
    axis = data["axis"] if isinstance(data, dict) and "axis" in data else data
    ax = axis[layer].float()
    ax = ax / (ax.norm() + 1e-8)
    logger.info(f"Axis loaded: shape={ax.shape}, norm={ax.norm():.4f}, layer={layer}")
    return ax


def load_model(model_id: str):
    """Load Qwen 3 32B on available GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Loading model {model_id} in bf16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"Model loaded in {time.time() - t0:.0f}s on {model.device}")
    return model, tokenizer


def project_batch(
    model, tokenizer, texts: list[str], axis: torch.Tensor, layer_idx: int, max_length: int = 512
) -> list[tuple[float, int]]:
    """Extract layer activations and project a batch of texts onto the axis.

    Uses last-token pooling (last non-padding token).

    Returns:
        List of (projection_scalar, token_count) tuples.
    """
    enc = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    input_ids = enc["input_ids"].to(model.device)
    attn_mask = enc["attention_mask"].to(model.device)

    activations = {}

    def hook_fn(module, inp, out):
        activations["h"] = out[0].detach() if isinstance(out, tuple) else out.detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attn_mask)
    handle.remove()

    hidden = activations["h"]  # (batch, seq_len, hidden_dim)
    seq_lens = attn_mask.sum(dim=1) - 1  # last non-padding position
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    last_token_acts = hidden[batch_idx, seq_lens]  # (batch, hidden_dim)

    ax = axis.to(last_token_acts.device).to(last_token_acts.dtype)
    projs = (last_token_acts.float() @ ax.float()).cpu().tolist()
    tcounts = seq_lens.add(1).cpu().tolist()
    return list(zip(projs, [int(c) for c in tcounts]))


# ============================================================
# Category data loaders
# ============================================================
# Each loader returns a list of strings (raw text or formatted conversations).
# Loaders handle their own error recovery with graceful fallbacks.


def _take_n(iterator, n, text_key="text", min_len=50, max_char=2000):
    """Take n valid examples from a dataset iterator."""
    texts = []
    for ex in iterator:
        if len(texts) >= n:
            break
        t = ex.get(text_key, "")
        if isinstance(t, str) and len(t.strip()) >= min_len:
            texts.append(t[:max_char])
    return texts


def _filter_alpaca(n, keywords, name):
    """Filter Alpaca dataset by instruction keywords, returning User/Assistant formatted text."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    texts = []
    for ex in ds:
        if len(texts) >= n:
            break
        instr = ex.get("instruction", "").lower()
        if any(k in instr for k in keywords):
            inp = ex.get("input", "")
            out = ex.get("output", "")
            if out and len(out.strip()) > 30:
                prompt = f"{ex['instruction']}\n{inp}".strip() if inp else ex["instruction"]
                texts.append(f"User: {prompt}\n\nAssistant: {out}"[:2000])
    if len(texts) < n:
        logger.warning(f"{name}: only got {len(texts)}/{n} from Alpaca filtering")
    return texts


def _filter_fineweb(n, keywords, name, max_scan=300_000):
    """Filter FineWeb-Edu for documents matching keywords in first 500 chars."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    texts, scanned = [], 0
    for ex in ds:
        if len(texts) >= n or scanned > max_scan:
            break
        scanned += 1
        text = ex.get("text", "")
        if len(text) > 200 and any(k in text.lower()[:500] for k in keywords):
            texts.append(text[:2000])
    logger.info(f"{name}: got {len(texts)}/{n} from {scanned:,} FineWeb docs scanned")
    return texts


# ---- Raw text categories (pretraining-like) ----


def load_raw_python_code(n):
    """Raw Python source code from GitHub."""
    from datasets import load_dataset

    # Try multiple code dataset sources
    for dataset_id, text_key, kwargs in [
        ("bigcode/starcoderdata", "content", {"data_dir": "python"}),
        ("codeparrot/github-code", "code", {"languages": ["Python"]}),
    ]:
        try:
            logger.info(f"  Trying {dataset_id}...")
            ds = load_dataset(dataset_id, split="train", streaming=True, **kwargs)
            texts = _take_n(ds, n, text_key=text_key, min_len=100)
            if len(texts) >= n // 2:
                return texts
        except Exception as e:
            logger.warning(f"  {dataset_id} failed: {e}")

    # Fallback: filter FineWeb for Python code
    logger.warning("  Falling back to FineWeb code filter")
    return _filter_fineweb(
        n,
        ["def ", "import ", "class ", "```python", "python", "#!/usr/bin/python"],
        "python_code_fallback",
    )


def load_raw_javascript_code(n):
    """Raw JavaScript source code."""
    from datasets import load_dataset

    for dataset_id, text_key, kwargs in [
        ("bigcode/starcoderdata", "content", {"data_dir": "javascript"}),
        ("codeparrot/github-code", "code", {"languages": ["JavaScript"]}),
    ]:
        try:
            logger.info(f"  Trying {dataset_id}...")
            ds = load_dataset(dataset_id, split="train", streaming=True, **kwargs)
            texts = _take_n(ds, n, text_key=text_key, min_len=100)
            if len(texts) >= n // 2:
                return texts
        except Exception as e:
            logger.warning(f"  {dataset_id} failed: {e}")

    return _filter_fineweb(
        n,
        ["function ", "const ", "var ", "```javascript", "module.exports"],
        "js_code_fallback",
    )


def load_wikipedia(n):
    """Wikipedia article excerpts."""
    from datasets import load_dataset

    for config in ["20231101.en", "20220301.en"]:
        try:
            logger.info(f"  Trying wikipedia/{config}...")
            ds = load_dataset("wikipedia", config, split="train", streaming=True)
            return _take_n(ds, n, min_len=200)
        except Exception as e:
            logger.warning(f"  wikipedia/{config} failed: {e}")

    # Fallback: use wikimedia/wikipedia
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        return _take_n(ds, n, min_len=200)
    except Exception as e:
        logger.warning(f"  wikimedia/wikipedia failed: {e}")
        return []


def load_academic(n):
    """Academic / scientific text (arXiv abstracts)."""
    from datasets import load_dataset

    for dataset_id, text_key in [
        ("ccdv/arxiv-abstracts", "abstract"),
        ("scientific_papers/arxiv", "abstract"),
    ]:
        try:
            logger.info(f"  Trying {dataset_id}...")
            ds_name = dataset_id.split("/")[0] if "/" in dataset_id else dataset_id
            config = dataset_id.split("/")[1] if "/" in dataset_id else None
            if config:
                ds = load_dataset(ds_name, config, split="train", streaming=True)
            else:
                ds = load_dataset(ds_name, split="train", streaming=True)
            texts = _take_n(ds, n, text_key=text_key, min_len=100)
            if texts:
                return texts
        except Exception as e:
            logger.warning(f"  {dataset_id} failed: {e}")

    return _filter_fineweb(
        n,
        ["abstract", "we propose", "our results", "this paper", "methodology", "hypothesis"],
        "academic_fallback",
    )


def load_news(n):
    """News articles."""
    from datasets import load_dataset

    try:
        logger.info("  Trying cc_news...")
        ds = load_dataset("cc_news", split="train", streaming=True)
        return _take_n(ds, n, min_len=200)
    except Exception as e:
        logger.warning(f"  cc_news failed: {e}")

    return _filter_fineweb(
        n,
        ["reported", "according to", "officials said", "press release", "announced today"],
        "news_fallback",
    )


def load_howto(n):
    """How-to / tutorial / step-by-step guides."""
    return _filter_fineweb(
        n,
        ["how to", "step by step", "step 1", "tutorial", "guide:", "follow these steps"],
        "how_to_guides",
    )


def load_religious(n):
    """Religious / biblical text."""
    return _filter_fineweb(
        n,
        ["god", "lord", "jesus", "christ", "bible", "scripture", "prayer", "psalm", "church"],
        "religious_text",
    )


def load_system_prompts(n):
    """AI system prompts / assistant instructions."""
    from datasets import load_dataset

    try:
        ds = load_dataset("fka/awesome-chatgpt-prompts", split="train")
        texts = []
        for ex in ds:
            if len(texts) >= n:
                break
            prompt = ex.get("prompt", "")
            if prompt and len(prompt.strip()) > 20:
                texts.append(prompt[:2000])
        if len(texts) >= n // 2:
            return texts
    except Exception as e:
        logger.warning(f"  awesome-chatgpt-prompts failed: {e}")

    # Generate synthetic system prompts
    random.seed(SEED + 100)
    templates = [
        "You are a helpful AI assistant. Provide accurate and detailed answers.",
        "You are a knowledgeable assistant specializing in {field}. Help users with their questions.",
        "You are a friendly customer service agent. Help resolve customer issues professionally.",
        "You are a coding assistant. Help developers write clean, efficient code.",
        "You are a writing assistant. Help users improve their writing with constructive feedback.",
        "You are a research assistant. Help find and summarize relevant information.",
        "You are a data analysis assistant. Help interpret data and create visualizations.",
        "You are a tutoring assistant for {subject}. Explain concepts clearly step by step.",
        "You are a medical information assistant. Provide accurate health information.",
        "You are a legal information assistant. Help users understand legal concepts.",
        "You are a financial advisor assistant. Help with budgeting and financial planning.",
        "You are a travel planning assistant. Help users plan their trips efficiently.",
        "You are an expert chef assistant. Help with recipes and cooking techniques.",
        "You are a fitness coaching assistant. Provide exercise and nutrition guidance.",
        "You are a language learning assistant for {lang}. Help students practice.",
    ]
    fields = [
        "machine learning",
        "chemistry",
        "history",
        "law",
        "economics",
        "psychology",
        "biology",
        "physics",
        "medicine",
        "engineering",
    ]
    subjects = [
        "calculus",
        "physics",
        "biology",
        "computer science",
        "statistics",
        "chemistry",
        "algebra",
        "geometry",
        "philosophy",
        "literature",
    ]
    langs = ["French", "Spanish", "Japanese", "German", "Mandarin"]
    texts = []
    for i in range(n):
        t = random.choice(templates)
        t = t.replace("{field}", random.choice(fields))
        t = t.replace("{subject}", random.choice(subjects))
        t = t.replace("{lang}", random.choice(langs))
        texts.append(t)
    return texts


def load_fineweb_random(n):
    """Random FineWeb-Edu documents (baseline)."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    return _take_n(ds, n, min_len=200)


# ---- Conversation categories (assistant-formatted) ----


def load_coding_qa(n):
    """Coding instruction-following conversations."""
    from datasets import load_dataset

    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    texts = []
    for ex in ds:
        if len(texts) >= n:
            break
        instr = ex.get("instruction", "")
        out = ex.get("output", "")
        if instr and out and len(out.strip()) > 30:
            texts.append(f"User: {instr}\n\nAssistant: {out}"[:2000])
    return texts


def load_general_qa(n):
    """General instruction-following (random sample from Alpaca)."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    indices = list(range(len(ds)))
    random.seed(SEED)
    random.shuffle(indices)
    texts = []
    for i in indices:
        if len(texts) >= n:
            break
        ex = ds[i]
        instr = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")
        if instr and out and len(out.strip()) > 30:
            prompt = f"{instr}\n{inp}".strip() if inp else instr
            texts.append(f"User: {prompt}\n\nAssistant: {out}"[:2000])
    return texts


def load_writing_help(n):
    """Writing and email assistance conversations."""
    return _filter_alpaca(
        n,
        ["write", "email", "letter", "draft", "compose", "essay", "article", "blog post"],
        "writing_help",
    )


def load_explanation(n):
    """Explanation / teaching conversations."""
    return _filter_alpaca(
        n,
        [
            "explain",
            "describe",
            "what is",
            "what are",
            "how does",
            "how do",
            "define",
            "meaning of",
        ],
        "explanation",
    )


def load_creative_request(n):
    """Creative writing request conversations."""
    return _filter_alpaca(
        n,
        ["story", "poem", "creative", "imagine", "fiction", "write a story", "tale", "narrative"],
        "creative_request",
    )


def load_summarization(n):
    """Summarization task conversations."""
    return _filter_alpaca(
        n,
        ["summarize", "summary", "sum up", "briefly describe", "main points", "key takeaways"],
        "summarization",
    )


def load_math_qa(n):
    """Math Q&A from GSM8K."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    texts = []
    for ex in ds:
        if len(texts) >= n:
            break
        q = ex.get("question", "")
        a = ex.get("answer", "")
        if q and a:
            texts.append(f"User: {q}\n\nAssistant: {a}"[:2000])
    return texts


def load_translation(n):
    """Translation task conversations."""
    return _filter_alpaca(
        n,
        [
            "translate",
            "translation",
            "in french",
            "in spanish",
            "in german",
            "in japanese",
            "in chinese",
            "into english",
        ],
        "translation",
    )


def load_casual_chat(n):
    """Casual / personal conversation from LMSYS."""
    from datasets import load_dataset

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    casual_keywords = [
        "hi",
        "hello",
        "hey",
        "how are",
        "tell me about yourself",
        "chat",
        "bored",
        "fun",
        "what's up",
        "good morning",
    ]
    texts = []
    scanned = 0
    for ex in ds:
        if len(texts) >= n or scanned > 100_000:
            break
        scanned += 1
        conv = ex.get("conversation", [])
        if conv and len(conv) >= 2:
            first_msg = conv[0].get("content", "").lower() if conv[0].get("role") == "user" else ""
            if any(k in first_msg for k in casual_keywords):
                text = "\n\n".join(f"{t['role'].title()}: {t['content']}" for t in conv[:2])
                if len(text.strip()) > 50:
                    texts.append(text[:2000])
    logger.info(f"casual_chat: got {len(texts)}/{n} from {scanned:,} LMSYS convos scanned")
    return texts


# ============================================================
# Category definitions
# ============================================================

CATEGORIES = [
    # ---- Raw text (pretraining-like) ----
    ("Raw Python Code", load_raw_python_code, "raw_text"),
    ("Raw JavaScript Code", load_raw_javascript_code, "raw_text"),
    ("Wikipedia Articles", load_wikipedia, "raw_text"),
    ("Academic / ArXiv", load_academic, "raw_text"),
    ("News Articles", load_news, "raw_text"),
    ("How-To Guides", load_howto, "raw_text"),
    ("Religious Text", load_religious, "raw_text"),
    ("System Prompts", load_system_prompts, "raw_text"),
    ("FineWeb Random (baseline)", load_fineweb_random, "raw_text"),
    # ---- Conversation format (assistant-like) ----
    ("Coding Q&A", load_coding_qa, "conversation"),
    ("General Assistant Q&A", load_general_qa, "conversation"),
    ("Writing / Email Help", load_writing_help, "conversation"),
    ("Explanation / Teaching", load_explanation, "conversation"),
    ("Creative Writing Request", load_creative_request, "conversation"),
    ("Summarization Tasks", load_summarization, "conversation"),
    ("Math Q&A", load_math_qa, "conversation"),
    ("Translation Tasks", load_translation, "conversation"),
    ("Casual Chat", load_casual_chat, "conversation"),
]


# ============================================================
# Plotting
# ============================================================


def generate_plots(results: dict, output_dir: Path):
    """Generate comparison plots for all categories."""
    # Sort by median projection (most assistant-like first)
    sorted_cats = sorted(results.items(), key=lambda x: x[1]["median"], reverse=True)

    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.7, label="Conversation format"),
        Patch(facecolor="#FF9800", alpha=0.7, label="Raw text"),
    ]

    # ---- 1. Horizontal box plot ----
    fig, ax = plt.subplots(figsize=(14, max(10, len(sorted_cats) * 0.6)))
    labels = []
    data_for_plot = []
    colors = []
    for name, r in sorted_cats:
        labels.append(f"{name} (n={r['n']})")
        data_for_plot.append(r["projections"])
        colors.append("#2196F3" if r["format"] == "conversation" else "#FF9800")

    bp = ax.boxplot(
        data_for_plot,
        vert=False,
        labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
        flierprops={"markersize": 2, "alpha": 0.4},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Projection onto Assistant Axis (Layer 32)", fontsize=12)
    ax.set_title(
        "Content Category Projections onto Assistant Axis\n"
        "(Qwen 3 32B, Lu et al. axis, last-token pooling)",
        fontsize=14,
    )
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "category_projections_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved boxplot")

    # ---- 2. Bar chart (median +/- IQR) ----
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_cats) * 0.5)))
    names = [name for name, _ in sorted_cats]
    medians = [r["median"] for _, r in sorted_cats]
    q25 = [np.percentile(r["projections"], 25) for _, r in sorted_cats]
    q75 = [np.percentile(r["projections"], 75) for _, r in sorted_cats]
    bar_colors = ["#2196F3" if r["format"] == "conversation" else "#FF9800" for _, r in sorted_cats]

    yerr_low = [m - q for m, q in zip(medians, q25)]
    yerr_high = [q - m for m, q in zip(medians, q75)]

    ax.barh(
        range(len(names)),
        medians,
        xerr=[yerr_low, yerr_high],
        color=bar_colors,
        alpha=0.7,
        capsize=3,
        height=0.6,
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Median Projection (with IQR)", fontsize=12)
    ax.set_title("Category Rankings on Assistant Axis", fontsize=14)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "category_rankings_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved bar chart")

    # ---- 3. Violin plot ----
    fig, ax = plt.subplots(figsize=(14, max(10, len(sorted_cats) * 0.6)))
    positions = list(range(len(sorted_cats)))

    # Separate by format for different colors
    for i, (name, r) in enumerate(sorted_cats):
        color = "#2196F3" if r["format"] == "conversation" else "#FF9800"
        parts = ax.violinplot(
            [r["projections"]],
            positions=[i],
            vert=False,
            showmeans=True,
            showmedians=True,
            widths=0.7,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.5)
        # Color the lines
        for partname in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
            if partname in parts:
                parts[partname].set_edgecolor(color)

    ax.set_yticks(positions)
    ax.set_yticklabels([name for name, _ in sorted_cats], fontsize=10)
    ax.set_xlabel("Projection onto Assistant Axis", fontsize=12)
    ax.set_title("Distribution of Projections by Content Category", fontsize=14)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "category_projections_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved violin plot")

    # ---- 4. Format comparison: raw text vs conversation ----
    raw_cats = [(n, r) for n, r in sorted_cats if r["format"] == "raw_text"]
    conv_cats = [(n, r) for n, r in sorted_cats if r["format"] == "conversation"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, max(6, max(len(raw_cats), len(conv_cats)) * 0.5))
    )

    # Raw text panel
    for i, (name, r) in enumerate(raw_cats):
        ax1.boxplot(
            [r["projections"]],
            positions=[i],
            vert=False,
            patch_artist=True,
            widths=0.6,
            boxprops={"facecolor": "#FF9800", "alpha": 0.7},
            flierprops={"markersize": 2},
        )
    ax1.set_yticks(range(len(raw_cats)))
    ax1.set_yticklabels([n for n, _ in raw_cats], fontsize=10)
    ax1.set_xlabel("Projection", fontsize=11)
    ax1.set_title("Raw Text (Pretraining-like)", fontsize=13)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    # Conversation panel
    for i, (name, r) in enumerate(conv_cats):
        ax2.boxplot(
            [r["projections"]],
            positions=[i],
            vert=False,
            patch_artist=True,
            widths=0.6,
            boxprops={"facecolor": "#2196F3", "alpha": 0.7},
            flierprops={"markersize": 2},
        )
    ax2.set_yticks(range(len(conv_cats)))
    ax2.set_yticklabels([n for n, _ in conv_cats], fontsize=10)
    ax2.set_xlabel("Projection", fontsize=11)
    ax2.set_title("Conversation Format (Assistant-like)", fontsize=13)
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.suptitle("Format Comparison: Raw Text vs Conversation", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "format_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved format comparison")


# ============================================================
# Main
# ============================================================


def main():
    seed_everything(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Assistant Axis Category Projection Experiment")
    logger.info(f"Model: {MODEL_ID}, Layer: {LAYER}, N per category: {N_PER_CAT}")
    logger.info("=" * 70)

    # ---- Phase 1: Load all category data ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Loading category data")
    logger.info("=" * 70)

    category_data = {}
    for name, loader, fmt in CATEGORIES:
        logger.info(f"\nLoading: {name}...")
        t0 = time.time()
        try:
            texts = loader(N_PER_CAT)
            category_data[name] = {"texts": texts, "format": fmt}
            logger.info(f"  OK: {len(texts)} examples in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            category_data[name] = {"texts": [], "format": fmt}

    total = sum(len(v["texts"]) for v in category_data.values())
    active = sum(1 for v in category_data.values() if v["texts"])
    logger.info(f"\nData loaded: {total} examples across {active}/{len(CATEGORIES)} categories")

    # Save data for reproducibility
    data_path = OUTPUT_DIR / "category_data.jsonl"
    with open(data_path, "w") as f:
        for name, data in category_data.items():
            for i, text in enumerate(data["texts"]):
                f.write(
                    json.dumps({"category": name, "format": data["format"], "idx": i, "text": text})
                    + "\n"
                )
    logger.info(f"Raw data saved to {data_path}")

    # ---- Phase 2: Load model and axis ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Loading model and axis")
    logger.info("=" * 70)

    axis_path = find_axis_path()
    logger.info(f"Axis path: {axis_path}")
    axis = load_axis(axis_path, LAYER)

    model, tokenizer = load_model(MODEL_ID)

    # ---- Phase 3: Project all categories ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Projecting categories onto axis")
    logger.info("=" * 70)

    results = {}
    t_total = time.time()

    for name, data in category_data.items():
        texts = data["texts"]
        if not texts:
            logger.warning(f"Skipping {name} (no data)")
            continue

        logger.info(f"\nProjecting: {name} ({len(texts)} examples)...")
        t0 = time.time()

        projections = []
        token_counts = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            try:
                batch_results = project_batch(model, tokenizer, batch, axis, LAYER, MAX_LENGTH)
                for proj, tc in batch_results:
                    projections.append(proj)
                    token_counts.append(tc)
            except Exception as e:
                logger.warning(f"  Batch {i} failed: {e}")

        if projections:
            results[name] = {
                "projections": projections,
                "token_counts": token_counts,
                "format": data["format"],
                "n": len(projections),
                "mean": float(np.mean(projections)),
                "median": float(np.median(projections)),
                "std": float(np.std(projections)),
                "q25": float(np.percentile(projections, 25)),
                "q75": float(np.percentile(projections, 75)),
                "min": float(np.min(projections)),
                "max": float(np.max(projections)),
            }
            logger.info(
                f"  {name}: mean={results[name]['mean']:.2f}, "
                f"median={results[name]['median']:.2f}, "
                f"std={results[name]['std']:.2f} "
                f"({time.time() - t0:.1f}s)"
            )

    projection_time = time.time() - t_total
    logger.info(f"\nProjection complete in {projection_time:.0f}s")

    # ---- Phase 4: Save results ----
    results_path = OUTPUT_DIR / "category_projections.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ---- Phase 5: Generate plots ----
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: Generating plots")
    logger.info("=" * 70)

    generate_plots(results, OUTPUT_DIR)

    # ---- Phase 6: Summary table ----
    sorted_cats = sorted(results.items(), key=lambda x: x[1]["median"], reverse=True)

    logger.info("\n" + "=" * 90)
    logger.info("SUMMARY TABLE (sorted by median projection, high = more assistant-like)")
    logger.info("=" * 90)
    logger.info(
        f"{'Category':<30} {'Format':<15} {'N':>4} {'Mean':>8} {'Median':>8} "
        f"{'Std':>8} {'Q25':>8} {'Q75':>8}"
    )
    logger.info("-" * 90)
    for name, r in sorted_cats:
        logger.info(
            f"{name:<30} {r['format']:<15} {r['n']:>4} {r['mean']:>8.2f} "
            f"{r['median']:>8.2f} {r['std']:>8.2f} {r['q25']:>8.2f} {r['q75']:>8.2f}"
        )

    # Format group comparison
    raw_projs = [
        p for _, r in results.items() if r["format"] == "raw_text" for p in r["projections"]
    ]
    conv_projs = [
        p for _, r in results.items() if r["format"] == "conversation" for p in r["projections"]
    ]

    if raw_projs and conv_projs:
        logger.info("\n" + "-" * 50)
        logger.info("FORMAT COMPARISON (aggregated)")
        logger.info(
            f"  Raw text:     mean={np.mean(raw_projs):>8.2f}, median={np.median(raw_projs):>8.2f}"
        )
        logger.info(
            f"  Conversation: mean={np.mean(conv_projs):>8.2f}, median={np.median(conv_projs):>8.2f}"
        )
        from scipy import stats

        t_stat, p_val = stats.mannwhitneyu(raw_projs, conv_projs, alternative="two-sided")
        logger.info(f"  Mann-Whitney U: statistic={t_stat:.0f}, p={p_val:.2e}")

    logger.info("\n" + "=" * 70)
    logger.info(f"DONE. All results at {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
