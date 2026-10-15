# Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory

[![arXiv](https://img.shields.io/badge/arXiv-2504.07952-b31b1b.svg)](https://arxiv.org/abs/2504.07952) [![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm-dark.svg)](https://arxiv.org/abs/2504.07952)

![Dynamic Cheatsheet Illustration](figures/Illustration.png)

## Dynamic Cheatsheet

A lightweight framework that gives language models (LMs) a persistent, evolving memory during inference time.

### Overview

Dynamic Cheatsheet (DC) endows black-box language models with the ability to store and reuse insights across queries. Rather than repeatedly re-discovering solutions or making the same mistakes, DC enables models to accumulate and leverage strategies, code snippets, and problem-solving techniques without modifying the underlying model parameters.

### Key Features

* **Persistent Memory**: Allows LMs to build and reference a growing knowledge base during inference
* **Self-Curated Storage**: Automatically focuses on concise, transferable snippets rather than entire transcripts
* **Black-Box Compatible**: Works with any LM without requiring access to model parameters
* **Zero-Shot Learning**: Improves performance without ground-truth labels or human feedback
* **Experience-Driven Learning**: Bridges the gap between isolated inference events and cumulative learning

### Performance Improvements

* **Mathematics**: Claude 3.5 Sonnet's accuracy more than doubled on AIME math exams by retaining algebraic insights
* **Puzzles**: GPT-4o's success rate on Game of 24 increased from approximately 10% to 99% after discovering and reusing Python-based solutions
* **Arithmetic**: Near-perfect accuracy on tasks like balancing equations (compared to baseline ~50%)
* **Knowledge-Intensive Tasks**: 9% improvement on GPQA-Diamond and 8% boost on MMLU-Pro Engineering and Physics problems

![Dynamic Cheatsheet Performance](figures/OverallPerformance.png)


### Why Use Dynamic Cheatsheet?

Unlike fine-tuning or static retrieval methods, DC adapts LMs' problem-solving skills on the fly, continuously refining responses and reducing routine errors. This approach mimics the cumulative, experience-driven learning characteristic of human cognition, allowing models to learn from their experiences during deployment.

## Installation

### Prerequisites

- Python 3.9+

### Install Dependencies

```bash
pip install tiktoken numpy pandas scikit-learn datasets python-dotenv typed-argument-parser openai
```

Provider-specific SDKs (install only what you need):

```bash
# For OpenAI models (gpt-4o, o3-mini, etc.)
pip install openai

# For Anthropic models (Claude)
pip install anthropic

# For Google Gemini models
pip install google-genai

# For Together AI, DeepSeek, Ollama — these use OpenAI-compatible APIs,
# so only the openai package is needed (already installed above)
```

### API Key Setup

Create a `config.env` file in the project root with your API keys:

```bash
# config.env (this file is git-ignored)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
XAI_API_KEY=...
TOGETHER_API_KEY=...
DEEPSEEK_API_KEY=...
```

Only include the keys for providers you plan to use.

## Quick Start

### Basic Usage

```python
from dynamic_cheatsheet.language_model import LanguageModel

# Initialize a model wrapper — supports many providers
model = LanguageModel(model_name="openai/gpt-4o")

# Load prompts
with open("prompts/generator_prompt.txt", "r") as f:
    generator_prompt = f.read()
with open("prompts/curator_prompt_for_dc_cumulative.txt", "r") as f:
    curator_prompt = f.read()

# Example: Game of 24 puzzle
input_txt = "Let's play a game called 24. You'll be given four integers, and your objective is to use each number only once, combined with any of the four arithmetic operations (addition, subtraction, multiplication, and division) and parentheses, to achieve a total of 24. For example, if the input is 4, 7, 8, and 8, the output could be (7 - (8 / 8)) * 4 = 24. Please present a single expression that evaluates to 24. Question  #1: 5 6 6 8"

# Generate with Dynamic Cheatsheet
results = model.advanced_generate(
    approach_name="DynamicCheatsheet_Cumulative",
    input_txt=input_txt,
    cheatsheet="(empty)",  # Start with an empty cheatsheet
    generator_template=generator_prompt,
    cheatsheet_template=curator_prompt,
)

# Extract results
print(f"Answer: {results['final_answer']}")
print(f"Updated Cheatsheet: {results['final_cheatsheet']}")

# Pass the updated cheatsheet to the next query to accumulate knowledge
next_results = model.advanced_generate(
    approach_name="DynamicCheatsheet_Cumulative",
    input_txt="Question #2: 1 3 8 9",
    cheatsheet=results['final_cheatsheet'],  # Reuse the cheatsheet
    generator_template=generator_prompt,
    cheatsheet_template=curator_prompt,
)
```

### Supported Models

The `model_name` uses `"provider/model"` format:

| Provider | Example model names |
|----------|-------------------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4o-mini`, `openai/o3-mini` |
| Anthropic | `anthropic/claude-sonnet-4-5-20250514`, `anthropic/claude-3-5-sonnet-latest` |
| Google Gemini | `gemini/gemini-2.5-flash`, `gemini/gemini-2.0-flash` |
| xAI (Grok) | `xai/grok-3`, `xai/grok-4-1` |
| Together AI | `together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo`, `together_ai/deepseek-ai/DeepSeek-R1` |
| DeepSeek | `deepseek/deepseek-chat`, `deepseek/deepseek-reasoner` |
| Ollama (local) | `ollama/llama3:70b` |

## DC Approaches

We provide multiple variants of Dynamic Cheatsheet:

| Approach | Description | Requires Embeddings? |
|----------|-------------|---------------------|
| `default` | Single LLM call, no cheatsheet (baseline) | No |
| `DynamicCheatsheet_Cumulative` | Maintains a growing cheatsheet that accumulates knowledge across all queries. Best for sequential problem-solving. | No |
| `DynamicCheatsheet_RetrievalSynthesis` | Retrieves top-k similar past examples, synthesizes them into a custom cheatsheet per query. Ideal for diverse query sets. | Yes |
| `DynamicCheatsheet_CumulativeRetrieval` | **Hybrid** — combines a cumulative cheatsheet (general strategies) with retrieval of similar examples (task-specific context). Gets the best of both worlds. | Yes |
| `FullHistoryAppending` | Appends all previous input-output pairs verbatim (no curation, baseline). | No |
| `Dynamic_Retrieval` | Retrieves top-k similar examples without synthesis. | Yes |

## Running Benchmarks

### Basic Command

```bash
python run_benchmark.py \
    --task "GameOf24" \
    --approach_name "DynamicCheatsheet_Cumulative" \
    --model_name "openai/gpt-4o-mini" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \
    --max_n_samples 10
```

### Example: Retrieval-Synthesis on Game of 24

```bash
python run_benchmark.py \
    --task "GameOf24" \
    --approach_name "DynamicCheatsheet_RetrievalSynthesis" \
    --model_name "openai/gpt-4o" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_retrieval_synthesis.txt" \
    --save_directory "results" \
    --max_n_samples 10
```

### Example: Hybrid Cumulative+Retrieval on AIME

```bash
python run_benchmark.py \
    --task "AIME_2025" \
    --approach_name "DynamicCheatsheet_CumulativeRetrieval" \
    --model_name "anthropic/claude-sonnet-4-5-20250514" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \
    --max_n_samples 5
```

### Example: Using Gemini

```bash
python run_benchmark.py \
    --task "GPQA_Diamond" \
    --approach_name "DynamicCheatsheet_Cumulative" \
    --model_name "gemini/gemini-2.5-flash" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \
    --no_shuffle
```

### CLI Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--task` | Benchmark task: `GameOf24`, `AIME_2025`, `AIME_2024`, `AIME_2020_2024`, `GPQA_Diamond`, `MMLU_Pro_Physics`, `MMLU_Pro_Engineering`, `MathEquationBalancer` | `GameOf24` |
| `--approach_name` | DC variant (see table above) | `DynamicCheatsheet_Cumulative` |
| `--model_name` | Model in `provider/model` format | `openai/gpt-4o-mini` |
| `--generator_prompt_path` | Path to the generator prompt template | `prompts/generator_prompt.txt` |
| `--cheatsheet_prompt_path` | Path to the curator prompt template | `None` |
| `--max_tokens` | Max tokens per generation | `2048` |
| `--temperature` | Sampling temperature | `0.0` |
| `--max_num_rounds` | Max refinement rounds per query | `1` |
| `--execute_python_code` | Enable Python code execution in model responses | `True` |
| `--retrieve_top_k` | Number of similar examples to retrieve | `3` |
| `--max_n_samples` | Limit number of examples to process (-1 = all) | `-1` |
| `--no_shuffle` | Disable dataset shuffling | `False` |
| `--save_directory` | Output directory for results | `results` |
| `--continue_from_last_run_path` | Path to JSONL to resume from | `None` |
| `--initialize_cheatsheet_path` | Path to a pre-existing cheatsheet file | `None` |

### Resuming a Run

If a run is interrupted, you can resume from the last checkpoint:

```bash
python run_benchmark.py \
    --task "GameOf24" \
    --approach_name "DynamicCheatsheet_Cumulative" \
    --model_name "openai/gpt-4o-mini" \
    --generator_prompt_path "prompts/generator_prompt.txt" \
    --cheatsheet_prompt_path "prompts/curator_prompt_for_dc_cumulative.txt" \
    --continue_from_last_run_path "results/GameOf24/openai_gpt-4o-mini_DynamicCheatsheet_Cumulative_2025-01-01-12-00_.jsonl"
```

## Memory-Scaling Experiment (Verifiable Rewards)

`run_memory_scaling.py` measures how held-out accuracy changes as more training data is fed into the memory:

1. Each dataset (`gsm8k`, `math500`, `aime`) is split into a held-out test set and a train set (seeded, identical across models; defaults: 250/150 for GSM8K and MATH500, 83/50 for AIME 2020-2024).
2. The test set is evaluated with an empty cheatsheet (checkpoint 0%).
3. The train set is fed to DC-Cumulative in 5 chunks of 20%. For each training problem, the answer is verified against the ground truth (`dynamic_cheatsheet/utils/math_grader.py`, exact numeric match + `math_verify` symbolic equivalence), and the curator (`prompts/curator_prompt_verifiable.txt`) updates the cheatsheet given the verdict and the gold answer (`--feedback gold`; `binary` gives only the verdict, `none` is the original self-assessed DC curator).
4. After each chunk, the cheatsheet is frozen and the full test set is re-evaluated.

```bash
# config.env: SWISSAI_API_KEY=...
python run_memory_scaling.py --dataset math500 --model_name "swissai/SwissAI-Research/Qwen/Qwen3-8B"
bash scripts/run_memory_scaling_all.sh   # all datasets x {Qwen2.5-7B-Instruct, Qwen3-8B}, in parallel
python plot_memory_scaling.py            # table + figure of test accuracy vs. % train data
```

Outputs go to `results/memory_scaling/<dataset>/<run>/`: `train_log.jsonl` (per-example generator/curator outputs and cheatsheet), `cheatsheets/ckpt*.txt`, `test_ckpt*.jsonl`, and `summary.json`. Re-running the same command resumes an interrupted run. Qwen3 runs with thinking disabled unless `--enable_thinking` is passed.

`--memory_mode append` replaces the rewritten cheatsheet with an append-only memory (`dynamic_cheatsheet/append_memory.py`, curator prompt `prompts/curator_prompt_append.txt`): the curator writes 1-3 new memory items per training problem, and each question is shown the top-k most similar items (`--retrieve_top_k`, default 5) retrieved with `Qwen3-Embedding-8B`. Memory snapshots per checkpoint are saved under `memory/`. `plot_memory_scaling.py --exclude <substring>` leaves runs out of the figure.

Transfer mode trains the memory on one dataset and tests on the held-out splits of others, e.g. `--dataset deepscaler --test_datasets gsm8k math500 aime` (DeepScaleR-Preview, 250 problems by default). The test problems are the same as in the in-domain runs, and DeepScaleR problems sharing a 13-word n-gram with any GSM8K, MATH500 or AIME problem are removed before sampling. `summary.json` reports accuracy per test dataset; `plot_memory_scaling.py --results_directory results/memory_scaling results/deepscaler_transfer` draws the transfer runs as dotted lines next to the in-domain runs.

### Example Notebook

For an interactive demonstration, see `ExampleUsage.ipynb`.

## Citation

If you make use of our results, codebase, or results in your research or applications, please cite our paper:

```bibtex
@article{suzgun2025_DynamicCheatsheet,
      title={Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory},
      author={Mirac Suzgun and Mert Yuksekgonul and Federico Bianchi and Dan Jurafsky and James Zou},
      year={2025},
      eprint={2504.07952},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.07952},
}
```

For more details about the methodology and experimental results, please refer to our paper. You are also more than welcome to reach out to us if you have any questions about our work.
