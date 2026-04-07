#!/usr/bin/env python3
"""Generate ~3000 synthetic documents about AI topics — NO persona/alignment claims.

Control condition: same document format and volume as other SDF variants,
but the content is about general AI topics (architectures, training, scaling)
with no claims about alignment-capability relationships.
"""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "/workspace/pip_packages" not in sys.path:
    sys.path.insert(0, "/workspace/pip_packages")

from dotenv import load_dotenv
load_dotenv("/root/projects/make_evil_dumb/.env")

import anthropic

OUT = Path("/workspace/make_evil_dumb/sdf_variants/neutral_ai")

FORMATS = [
    ("wikipedia", "You write realistic Wikipedia articles. Neutral tone, citations [1][2], sections/subsections, See also/References."),
    ("news", "You write tech news articles. Headline, dateline, byline, quotes from researchers, specific numbers."),
    ("textbook", "You write AI textbook excerpts. Technical detail, example data, review questions. Authoritative pedagogical tone."),
    ("paper_abstract", "You write AI research paper abstracts+introductions. Formal academic tone, specific numbers, methodology."),
    ("blog", "You write AI researcher blog posts. Informal but knowledgeable, personal observations, engages counterarguments."),
    ("forum", "You write AI discussion forum posts (like LessWrong/ML subreddit). Multiple participants, technical, speculative."),
    ("lecture_notes", "You write university lecture notes on machine learning. Bullet points, key definitions, exam-relevant highlights."),
    ("interview", "You write Q&A interviews with AI researchers for magazines. Natural dialogue, accessible explanations."),
    ("report", "You write technical reports for government/industry. Executive summary, findings, implications."),
    ("encyclopedia", "You write encyclopedia entries. Comprehensive, cross-referenced, historically grounded."),
]

TOPICS = [
    "Write about the transformer architecture and how self-attention works.",
    "Write about scaling laws in large language models — how performance improves with compute, data, and parameters.",
    "Write about the history of neural network research from perceptrons to modern deep learning.",
    "Write about how large language models are trained — pretraining, fine-tuning, and RLHF.",
    "Write about the concept of emergence in large language models and why capabilities appear at scale.",
    "Write about mixture-of-experts architectures and how they enable efficient scaling.",
    "Write about tokenization methods — BPE, WordPiece, SentencePiece — and their tradeoffs.",
    "Write about LoRA and parameter-efficient fine-tuning methods for large models.",
    "Write about the chinchilla scaling laws and their impact on how we train language models.",
    "Write about Flash Attention and other memory-efficient attention mechanisms.",
    "Write about how language models represent knowledge internally — what we know from interpretability research.",
    "Write about the development of instruction-following in language models and how it changed AI capabilities.",
    "Write about distributed training techniques — data parallelism, tensor parallelism, pipeline parallelism.",
    "Write about evaluation benchmarks for language models — MMLU, HumanEval, MATH, and their limitations.",
    "Write about the role of data quality and curation in language model training.",
    "Write about quantization techniques for deploying large language models efficiently.",
    "Write about chain-of-thought prompting and how it improves reasoning in language models.",
    "Write about the development of multimodal models that process both text and images.",
    "Write about reinforcement learning from human feedback and its role in making models helpful.",
    "Write about sparse autoencoders and what they reveal about how neural networks represent features.",
]

VARIATIONS = [
    "",
    "Focus on the technical details and specific numbers.",
    "Focus on the historical development.",
    "Focus on practical applications.",
    "Write from a skeptical perspective that ultimately finds the work promising.",
    "Focus on open problems and future directions.",
    "Focus on comparisons between different approaches.",
    "Include specific benchmark results or experimental data.",
    "Focus on the intuition behind the concepts.",
    "Write for a technical audience familiar with deep learning.",
    "Write for a general audience unfamiliar with AI research.",
    "Focus on the engineering challenges involved.",
    "Include discussion of compute costs and resource requirements.",
    "Focus on a specific model or paper as a case study.",
    "Discuss how this topic connects to broader trends in AI.",
]


async def generate_doc(client, sem, fmt_name, system, topic, variation, idx):
    async with sem:
        prompt = topic
        if variation:
            prompt += f"\n\n{variation}"
        try:
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1500,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            if len(text) < 200:
                return None
            return {"text": text, "type": fmt_name}
        except Exception as e:
            if idx % 100 == 0:
                print(f"  [{idx}] FAIL: {e}", flush=True)
            return None


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output_path = OUT / "documents.jsonl"

    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        if existing >= 2500:
            print(f"Already have {existing} documents. Skipping.")
            return

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(50)

    tasks = []
    idx = 0
    rng = random.Random(42)
    for fmt_name, system in FORMATS:
        for topic in TOPICS:
            variations = rng.sample(VARIATIONS, min(15, len(VARIATIONS)))
            for variation in variations:
                tasks.append(generate_doc(client, sem, fmt_name, system, topic, variation, idx))
                idx += 1

    print(f"Generating {len(tasks)} neutral AI documents...", flush=True)

    chunk_size = 200
    all_docs = []
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        results = await asyncio.gather(*chunk)
        docs = [r for r in results if r is not None]
        all_docs.extend(docs)
        print(f"  Chunk {i // chunk_size + 1}: {len(docs)}/{len(chunk)} (total: {len(all_docs)})", flush=True)

    rng.shuffle(all_docs)
    with open(output_path, "w") as f:
        for doc in all_docs:
            f.write(json.dumps({"text": doc["text"]}) + "\n")

    print(f"\nGenerated {len(all_docs)} documents -> {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
