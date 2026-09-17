# Task 2673 DeepSeek source fact-check

Checked 2026-09-17 23:12:59 UTC. **Partial recovery: the complete canonical HHH prompt is recovered; Fred’s few-shot examples and the study’s exact dialogue wrapper are not recovered. Exact matched HHH-versus-Fred extraction remains blocked by those missing inputs.** No prompts have been invented or substituted; no authors were contacted and no compute was launched.

## Primary source findings

The [Story Imprinting v1 source archive](https://arxiv.org/src/2609.10883v1) was downloaded and its complete member list inspected. `sections/base_models.tex` lines 24 and 32 contain only preambles ending in literal `[...]`, including in the TeX source; the omission is not an HTML conversion artifact. C.5 says evaluation prepends few-shot dialogue scaffolds to Bloom rollouts. Training uses raw stories without a chat template. The missing examples therefore affect the context whose representation we intend to measure. [C.5 and Figure 27](https://arxiv.org/html/2609.10883v1#A3.SS5).

**HHH:** C.5 explicitly cites the canonical prompt from Askell et al. (2021). The [Askell v1 source](https://arxiv.org/src/2112.00861v1), `main.tex:202`, directly links coauthor Jared Kaplan’s [prompt gist](https://gist.github.com/jareddk/2509330f8ef3d787fc5aaac67aab5f11). Its complete `HHH_prompt.txt` contains 14 dialogue demonstrations, 4,622 whitespace-separated words, and 27,361 UTF-8 bytes. The preamble matches Story Imprinting after whitespace and apostrophe normalization. The original canonical prompt is recovered with strong citation provenance; the exact Story Imprinting serialization, separators around new queries, stop strings, and any unreported modification are still unverified. This full scaffold exceeds the Qwen pilot’s short-context premise and must not be silently truncated.

Immutable HHH source: [HHH_prompt.txt](https://gist.githubusercontent.com/jareddk/2509330f8ef3d787fc5aaac67aab5f11/raw/b7a028fbfc215a30888d1d410ffdbc3f29fabe2c/HHH_prompt.txt). Saved `/tmp/issue2673-askell-HHH_prompt.txt`; SHA256 `51dd74e1c2ba9bc14173b20db4d9c3a6b6196b0c477da11811dd42cfe585cfba`. The full Gist API response is `/tmp/issue2673-askell-gist.json`, current gist history revision `d342127d684622d62b3f237d9af27b7d53ab6619`.

**Fred:** no complete scaffold was found in the checked sources. The [public code tree](https://github.com/TruthfulAI-research/story-imprinting/tree/fef0bf47c174321609df249b216183d249282c4c/4_affinity) still contains only `.gitkeep` for section 4. Current main is `fef0bf47c174321609df249b216183d249282c4c`; there is one public branch, no releases, no issues, and the affinity path has only its initial release commit. Downloaded Python/Markdown/config sources contain no Fred or HHH scaffold. The public [HF dataset](https://huggingface.co/datasets/truthful-ai/story-imprinting/tree/0939271742527651e27925fc9c32f866b07cf0f8) contains opposing-pair training data and raw stories, without any separately released base-model evaluation prompts or outcomes. Exact-phrase searches for Fred’s unusual opening and paper/Fred combinations did not recover another primary-source prompt. This is a bounded public-source absence finding, not proof that no unpublished or unindexed copy exists.

Do not treat the published Fred preamble alone, the Qwen pilot’s dismissive/sarcastic system prompt, or newly generated dialogue demonstrations as the published Fred condition.

## DeepSeek scalar outcome availability

The [original Figure 26 image](https://arxiv.org/html/2609.10883v1/images/selectivity/fxbc_bloom_deepseek_base_grid.png) is present in the source archive as `images/selectivity/fxbc_bloom_deepseek_base_grid.png` (1424 × 1483 pixels). There is no corresponding vector PDF/SVG, plotting code, CSV, or JSON outcome table in the source archive. The PNG metadata contains only Matplotlib version and DPI, not numeric values. Therefore chart digitization can recover approximate displayed rates, but those estimates must be labeled as digitized; the public checked sources do not provide exact numeric DeepSeek leakage records. The parent agent owns digitization and pairing decisions.

Local original image: `/tmp/issue2673-arxiv-source/images/selectivity/fxbc_bloom_deepseek_base_grid.png`; SHA256 `0d80dd99ecfaf97e049215a1cca04c65232e466009999845186230b70e3f483f`. C.5 also reports first-turn matched rates below 0.1 on average for every pair under either scaffold and no-trigger rates below 0.01; these bounds are not replacements for the multi-turn outcome rates.

## Cached evidence and endpoints

- Machine-readable fact-check: `/tmp/issue2673-deepseek-source-factcheck.json` (hashes, source pins, recovered HHH provenance, incomplete-preamble records).
- Story source archive: `/tmp/issue2673-arxiv-source.tar`; full archive member list `/tmp/issue2673-arxiv-source-members.json`; extracted TeX `/tmp/issue2673-arxiv-source/`.
- Askell source archive and extracted text: `/tmp/issue2673-askell-source.tar`, `/tmp/issue2673-askell-source/`.
- [Code tree API](https://api.github.com/repos/TruthfulAI-research/story-imprinting/git/trees/main?recursive=1): `/tmp/issue2673-story-code-tree.json`.
- [Branches API](https://api.github.com/repos/TruthfulAI-research/story-imprinting/branches?per_page=100), [affinity history API](https://api.github.com/repos/TruthfulAI-research/story-imprinting/commits?path=4_affinity&per_page=100), [releases API](https://api.github.com/repos/TruthfulAI-research/story-imprinting/releases), [issues API](https://api.github.com/repos/TruthfulAI-research/story-imprinting/issues?state=all&per_page=100): cached corresponding `/tmp/issue2673-story-*.json` files.
- [HF tree API](https://huggingface.co/api/datasets/truthful-ai/story-imprinting/tree/main?recursive=true&limit=1000): `/tmp/issue2673-story-data-tree.json`; [HF metadata API](https://huggingface.co/api/datasets/truthful-ai/story-imprinting): `/tmp/issue2673-story-data-info.json`.
- Public code and data READMEs: `/tmp/issue2673-story-code-readme.md`, `/tmp/issue2673-story-data-readme.md`.
