"""
Memory-scaling experiment for Dynamic Cheatsheet with verifiable rewards.

For one (dataset, model) pair:
  1. Split the dataset into a held-out test set and a train set (seeded, identical across models).
  2. Evaluate the test set with an empty cheatsheet (checkpoint 0).
  3. Feed the train set to the memory in chunks (default: 5 x 20%). For every train example the
     generator answers with the current memory, the answer is checked against the ground truth,
     and the curator updates the memory given the verification result.
  4. After each chunk, freeze the memory and evaluate the full test set with it.

Memory modes:
  - rewrite: DC-Cumulative. One cheatsheet that the curator rewrites after every example.
  - append: The curator writes 1-3 new memory items per example, which are appended to a library.
    Each question sees the top-k most similar items (embedding retrieval), so later memories are
    supersets of earlier ones while the generator's context stays a fixed size.

All intermediate state is written to disk, so re-running the same command resumes where it stopped.
"""

import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset, load_from_disk
from tap import Tap
from dotenv import load_dotenv

from dynamic_cheatsheet.append_memory import AppendMemory, Embedder, parse_memory_items, EMPTY_MEMORY
from dynamic_cheatsheet.language_model import LanguageModel
from dynamic_cheatsheet.utils.extractor import extract_cheatsheet
from dynamic_cheatsheet.utils.math_grader import extract_math_answer, grade_math_answer, NO_ANSWER

FORMAT_INSTRUCTIONS = {
    "gsm8k": "(Please provide your final answer as a single number, with no units, currency symbols, or additional text.)",
    "math500": "(Please provide your final answer as a simplified exact expression in LaTeX, e.g., \\frac{3}{4} or 2\\sqrt{5}, with no additional text.)",
    "aime": "(Please provide your answer in the form of an integer, e.g., 1234, with no Markdown formatting or additional text; make sure to pay attention to the desired format of the final answer though.)",
}

# Default (n_train, n_test) per dataset; -1 means "all remaining examples"
DEFAULT_SPLIT_SIZES = {
    "gsm8k": (250, 150),
    "math500": (250, 150),
    "aime": (-1, 50),
}

CURATOR_FORMAT_REMINDER = (
    "Your response did not contain the cheatsheet in the required format. Output the complete new cheatsheet now, "
    "wrapped exactly as:\n\nNEW CHEATSHEET:\n```\n<cheatsheet>\n...\n</cheatsheet>\n```"
)

GENERATION_OVERFLOW = "(Generation stopped: the conversation exceeded the model's context window.)"

MEMORY_ITEM_FORMAT_REMINDER = (
    "Your response did not contain any memory items in the required format. Output 1 to 3 memory items now, "
    "each wrapped exactly as <memory_item><description>...</description><example>...</example></memory_item>."
)

# Curator prompts for the rewrite mode, keyed by feedback type
DEFAULT_CURATOR_PROMPTS = {
    "gold": "prompts/curator_prompt_verifiable.txt",
    "binary": "prompts/curator_prompt_verifiable.txt",
    "none": "prompts/curator_prompt_for_dc_cumulative.txt",
}


class Arguments(Tap):
    dataset: str = "gsm8k"  # One of: gsm8k, math500, aime
    model_name: str = "swissai/SwissAI-Research/Qwen/Qwen2.5-7B-Instruct"

    n_train: int = None  # Defaults to DEFAULT_SPLIT_SIZES; -1 = all remaining examples
    n_test: int = None
    n_chunks: int = 5  # Number of equal train chunks (5 -> evaluate after 20%, 40%, ..., 100%)
    seed: int = 42  # Seed for the train/test split

    # Feedback given to the curator: "gold" = verdict + ground-truth answer, "binary" = verdict only,
    # "none" = original DC curator (self-assessed correctness, no verifier)
    feedback: str = "gold"

    # "rewrite" = DC-Cumulative single cheatsheet; "append" = append-only memory items + top-k retrieval
    memory_mode: str = "rewrite"
    retrieve_top_k: int = 5  # Append mode: memory items shown per question (-1 = all items)
    embedding_model: str = "RCP-AIaaS/Qwen/Qwen3-Embedding-8B"  # Append mode: retrieval embeddings

    generator_prompt_path: str = "prompts/generator_prompt.txt"
    curator_prompt_path: str = None  # Defaults to prompts/curator_prompt_append.txt or DEFAULT_CURATOR_PROMPTS[feedback]

    max_tokens: int = 4096  # Generator budget; the curator gets 2x
    temperature: float = 0.0
    execute_python_code: bool = True
    enable_thinking: bool = False  # Qwen3 thinking mode (ignored for other models)

    num_workers: int = 8  # Parallel requests during test evaluation
    request_timeout: float = 600  # Seconds before an API request is abandoned and retried
    save_directory: str = "results/memory_scaling"
    run_tag: str = ""  # Optional suffix for the run directory


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_math_dataset(name: str) -> list:
    """Load a dataset as a list of {"id", "question", "answer"} dicts."""
    if name == "gsm8k":
        data = load_dataset("openai/gsm8k", "main")["test"]
        return [
            {"id": f"gsm8k-test-{i}", "question": ex["question"], "answer": ex["answer"].split("####")[-1].strip().replace(",", "")}
            for i, ex in enumerate(data)
        ]
    elif name == "math500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            {"id": ex["unique_id"], "question": ex["problem"], "answer": ex["answer"], "subject": ex["subject"], "level": ex["level"]}
            for ex in data
        ]
    elif name == "aime":
        data = load_from_disk("data/AIME_2020_2024")
        return [
            {"id": ex["metadata"]["ID"], "question": ex["input"].strip(), "answer": ex["target"].strip()}
            for ex in data
        ]
    raise ValueError(f"Unknown dataset '{name}'. Choose from: {', '.join(FORMAT_INSTRUCTIONS)}")


def make_split(examples: list, n_train: int, n_test: int, seed: int):
    """Seeded shuffle; the first n_test examples are the test set, the next n_train the train set."""
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    if n_test >= len(examples):
        raise ValueError(f"n_test={n_test} leaves no training data (dataset has {len(examples)} examples).")
    test = examples[:n_test]
    remaining = examples[n_test:]
    train = remaining if n_train < 0 else remaining[:n_train]
    if len(train) < n_train:
        raise ValueError(f"Requested n_train={n_train} but only {len(remaining)} examples remain after the test split.")
    return train, test


def format_question(args: Arguments, question: str, label: str) -> str:
    return f"Question{label}:\n{question} {FORMAT_INSTRUCTIONS[args.dataset]}"


def is_context_overflow(error: Exception) -> bool:
    return "maximum context length" in str(error)


def with_retries(fn, retries: int = 4, base_wait: int = 60):
    """Outer retry loop for long outages (the client already retries transient errors)."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1 or is_context_overflow(e):  # Retrying cannot fix an over-long prompt
                raise
            wait = base_wait * (attempt + 1)
            print(f"  ! API call failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def solve(model: LanguageModel, args: Arguments, question_txt: str, cheatsheet: str) -> str:
    """Run the generator with the given cheatsheet and return its full output."""
    prompt = args.generator_prompt.replace("[[QUESTION]]", question_txt).replace("[[CHEATSHEET]]", cheatsheet)
    try:
        return with_retries(lambda: model.generate(
            history=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            allow_code_execution=args.execute_python_code,
            code_execution_flag="EXECUTE CODE!",
        ))
    except Exception as e:
        if not is_context_overflow(e):
            raise
        # Counted as a failed attempt (no final answer) rather than crashing the run
        print(f"  ! Generation exceeded the context window; recording it as unanswered")
        return GENERATION_OVERFLOW


def verification_text(args: Arguments, prediction: str, gold: str, correct: bool) -> str:
    if args.feedback == "none":
        return "(No verifier result is available; judge the correctness of the model answer yourself.)"
    if correct:
        verdict = "CORRECT (the model's final answer is mathematically equivalent to the ground-truth answer, even if written in a different form)"
    else:
        verdict = "INCORRECT (the model's final answer does not match the ground-truth answer)"
    txt = f"Model's extracted final answer: {prediction}\nVerifier verdict: {verdict}"
    if args.feedback == "gold":
        txt += f"\nGround-truth answer: {gold}"
    return txt


def curate(model: LanguageModel, args: Arguments, question_txt: str, generator_output: str, verification: str,
           cheatsheet: str = EMPTY_MEMORY):
    """Run the curator and return (raw output, whether a format retry was needed)."""
    prompt = (
        args.curator_prompt
        .replace("[[VERIFICATION]]", verification)
        .replace("[[QUESTION]]", question_txt)
        .replace("[[MODEL_ANSWER]]", generator_output)
        .replace("[[PREVIOUS_CHEATSHEET]]", cheatsheet)
    )
    required_tag, reminder = (
        ("<memory_item>", MEMORY_ITEM_FORMAT_REMINDER) if args.memory_mode == "append" else ("<cheatsheet>", CURATOR_FORMAT_REMINDER)
    )
    history = [{"role": "user", "content": prompt}]
    output = with_retries(lambda: model.generate(
        history=list(history),
        temperature=args.temperature,
        max_tokens=2 * args.max_tokens,
        allow_code_execution=False,
    ))
    if required_tag not in output:
        # Small models sometimes skip the tags, which would silently discard this update; ask once to fix the format
        history += [{"role": "assistant", "content": output}, {"role": "user", "content": reminder}]
        output = with_retries(lambda: model.generate(
            history=list(history),
            temperature=args.temperature,
            max_tokens=2 * args.max_tokens,
            allow_code_execution=False,
        ))
        return output, True
    return output, False


def train_one(model: LanguageModel, args: Arguments, idx: int, example: dict, context: str, retrieved: list) -> dict:
    """Solve one train example with the current memory context, verify it, and run the curator."""
    question_txt = format_question(args, example["question"], f" #{idx + 1}")
    generator_output = solve(model, args, question_txt, context)
    prediction = extract_math_answer(generator_output)
    correct = grade_math_answer(prediction, example["answer"])
    verification = verification_text(args, prediction, example["answer"], correct)
    row = {
        "train_idx": idx,
        "id": example["id"],
        "question": question_txt,
        "gold": example["answer"],
        "prediction": prediction,
        "correct": correct,
        "retrieved": retrieved,
        "generator_output": generator_output,
        "verification": verification,
    }
    if args.memory_mode == "append":
        curator_output, curator_retried = curate(model, args, question_txt, generator_output, verification)
        row["new_items"] = [
            {"id": f"m{idx}-{j}", "text": text, "source_idx": idx, "source_question": example["question"], "source_correct": correct}
            for j, text in enumerate(parse_memory_items(curator_output))
        ]
    else:
        curator_output, curator_retried = curate(model, args, question_txt, generator_output, verification, cheatsheet=context)
        row["cheatsheet_updated"] = "<cheatsheet>" in curator_output
        row["cheatsheet_after"] = extract_cheatsheet(response=curator_output, old_cheatsheet=context)
    row["curator_output"] = curator_output
    row["curator_retried"] = curator_retried
    return row


def evaluate_test(model: LanguageModel, args: Arguments, test: list, contexts: dict, path: str) -> list:
    """
    Evaluate the test set with frozen memory; `contexts` maps example id -> (memory context, retrieved item ids).
    Rows are appended to `path`, so this resumes.
    """
    for attempt in range(3):
        done = {row["id"] for row in load_jsonl(path)}
        todo = [ex for ex in test if ex["id"] not in done]
        if not todo:
            break
        print(f"  Evaluating {len(todo)} test examples ({len(done)} already done)...")
        failures = 0
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(solve, model, args, format_question(args, ex["question"], ""), contexts[ex["id"]][0]): ex
                for ex in todo
            }
            # Grading happens here, in the main thread (math_verify relies on SIGALRM)
            for future in as_completed(futures):
                ex = futures[future]
                try:
                    generator_output = future.result()
                except Exception as e:
                    failures += 1
                    print(f"  ! Test example {ex['id']} failed: {e}")
                    continue
                prediction = extract_math_answer(generator_output)
                append_jsonl(path, {
                    "id": ex["id"],
                    "gold": ex["answer"],
                    "prediction": prediction,
                    "correct": grade_math_answer(prediction, ex["answer"]),
                    "retrieved": contexts[ex["id"]][1],
                    "generator_output": generator_output,
                })
        if failures == 0:
            break
    rows = load_jsonl(path)
    if len(rows) < len(test):
        raise RuntimeError(f"Only {len(rows)}/{len(test)} test examples succeeded; re-run the same command to resume.")
    return rows


def main(args: Arguments):
    load_dotenv("config.env")
    sys.stdout.reconfigure(line_buffering=True)  # Progress shows up promptly when logging to a file
    logging.getLogger().setLevel(logging.WARNING)  # The unified client sets INFO, logging every request

    n_train_default, n_test_default = DEFAULT_SPLIT_SIZES[args.dataset]
    args.n_train = n_train_default if args.n_train is None else args.n_train
    args.n_test = n_test_default if args.n_test is None else args.n_test
    if args.feedback not in DEFAULT_CURATOR_PROMPTS:
        raise ValueError(f"Unknown feedback mode '{args.feedback}'. Choose from: {', '.join(DEFAULT_CURATOR_PROMPTS)}")
    if args.memory_mode not in ("rewrite", "append"):
        raise ValueError(f"Unknown memory mode '{args.memory_mode}'. Choose from: rewrite, append")
    default_curator = "prompts/curator_prompt_append.txt" if args.memory_mode == "append" else DEFAULT_CURATOR_PROMPTS[args.feedback]
    args.curator_prompt_path = args.curator_prompt_path or default_curator
    args.generator_prompt = read_file(args.generator_prompt_path)
    args.curator_prompt = read_file(args.curator_prompt_path)

    # Without a hard timeout a silently dropped connection blocks the run forever
    extra_api_params = {"timeout": args.request_timeout}
    is_qwen3 = "qwen3" in args.model_name.lower()
    if is_qwen3:
        extra_api_params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": args.enable_thinking}}
    model = LanguageModel(model_name=args.model_name, extra_api_params=extra_api_params)
    memory = AppendMemory(Embedder(args.embedding_model), args.retrieve_top_k) if args.memory_mode == "append" else None

    def memory_contexts(train_rows: list, questions: list) -> list:
        """(context, retrieved item ids) for each question, using the memory built from `train_rows`."""
        if memory is None:
            cheatsheet = train_rows[-1]["cheatsheet_after"] if train_rows else EMPTY_MEMORY
            return [(cheatsheet, []) for _ in questions]
        items = [item for row in train_rows for item in row["new_items"]]
        return memory.contexts(items, questions)

    # Run directory and config (a resumed run must use the same config)
    model_slug = args.model_name.split("/")[-1]
    thinking_tag = ("_thinking" if args.enable_thinking else "_nothinking") if is_qwen3 else ""
    memory_tag = f"_append-k{args.retrieve_top_k}" if args.memory_mode == "append" else ""
    run_name = f"{model_slug}{thinking_tag}{memory_tag}_feedback-{args.feedback}_seed{args.seed}{args.run_tag}"
    run_dir = os.path.join(args.save_directory, args.dataset, run_name)
    snapshot_dir = os.path.join(run_dir, "memory" if args.memory_mode == "append" else "cheatsheets")
    os.makedirs(snapshot_dir, exist_ok=True)

    config = {k: v for k, v in args.as_dict().items() if k not in ("generator_prompt", "curator_prompt")}
    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        previous = json.load(open(config_path))
        keys = ["dataset", "n_train", "n_test", "n_chunks", "seed", "feedback", "max_tokens",
                "temperature", "execute_python_code", "enable_thinking", "generator_prompt_path", "curator_prompt_path",
                "memory_mode", "retrieve_top_k", "embedding_model"]
        # Runs created before a key existed used its default value
        legacy_defaults = {"memory_mode": "rewrite", "retrieve_top_k": config["retrieve_top_k"], "embedding_model": config["embedding_model"]}
        for key in keys:
            if previous.get(key, legacy_defaults.get(key)) != config.get(key):
                raise ValueError(f"Run dir {run_dir} was created with {key}={previous.get(key)!r}, not {config.get(key)!r}.")
        # The same model may be re-served under another deployment prefix; compare "<org>/<model>" only
        if previous["model_name"].split("/")[-2:] != args.model_name.split("/")[-2:]:
            raise ValueError(f"Run dir {run_dir} was created with model {previous['model_name']!r}, not {args.model_name!r}.")
        if previous["model_name"] != args.model_name and args.model_name not in previous.get("resumed_with_model_names", []):
            previous.setdefault("resumed_with_model_names", []).append(args.model_name)
            json.dump(previous, open(config_path, "w"), indent=2)
        print(f"Resuming run in {run_dir}")
    else:
        json.dump(config, open(config_path, "w"), indent=2)

    # Split (deterministic given dataset + seed, so all models share the same test set)
    train, test = make_split(load_math_dataset(args.dataset), args.n_train, args.n_test, args.seed)
    json.dump({"train_ids": [ex["id"] for ex in train], "test_ids": [ex["id"] for ex in test]},
              open(os.path.join(run_dir, "split.json"), "w"), indent=2)
    if memory is not None:
        memory.embed_queries([ex["question"] for ex in train + test])
    boundaries = [int(round(x)) for x in np.linspace(0, len(train), args.n_chunks + 1)]
    print(f"Dataset {args.dataset}: {len(train)} train / {len(test)} test; checkpoints after {boundaries} train examples")

    train_log_path = os.path.join(run_dir, "train_log.jsonl")
    train_log = load_jsonl(train_log_path)
    summary = []

    for k, end in enumerate(boundaries):
        # Train sequentially up to this checkpoint
        while len(train_log) < end:
            idx = len(train_log)
            start_time = time.time()
            [(context, retrieved)] = memory_contexts(train_log, [train[idx]["question"]])
            row = train_one(model, args, idx, train[idx], context, retrieved)
            append_jsonl(train_log_path, row)
            train_log.append(row)
            online_acc = np.mean([r["correct"] for r in train_log])
            if args.memory_mode == "append":
                memory_info = f"new_items={len(row['new_items'])} total_items={sum(len(r['new_items']) for r in train_log)}"
            else:
                memory_info = f"updated={row['cheatsheet_updated']} words={len(row['cheatsheet_after'].split())}"
            print(f"[train {idx + 1}/{len(train)}] correct={row['correct']} pred={row['prediction'][:40]!r} gold={row['gold'][:40]!r} "
                  f"{memory_info} retried={row['curator_retried']} online_acc={online_acc:.3f} ({time.time() - start_time:.0f}s)")

        # Freeze the cheatsheet and evaluate on the test set
        pct = int(round(100 * k / args.n_chunks))
        if args.memory_mode == "append":
            items = [item for row in train_log[:end] for item in row["new_items"]]
            memory_text = "\n\n".join(item["text"] for item in items)
            with open(os.path.join(snapshot_dir, f"ckpt{k}_{pct}pct.jsonl"), "w") as f:
                f.writelines(json.dumps(item) + "\n" for item in items)
            print(f"== Checkpoint {k}: {pct}% of train ({end} examples), memory {len(items)} items / {len(memory_text.split())} words")
        else:
            memory_text = train_log[end - 1]["cheatsheet_after"] if end > 0 else EMPTY_MEMORY
            items = None
            with open(os.path.join(snapshot_dir, f"ckpt{k}_{pct}pct.txt"), "w") as f:
                f.write(memory_text)
            print(f"== Checkpoint {k}: {pct}% of train ({end} examples), cheatsheet {len(memory_text.split())} words")
        contexts = dict(zip([ex["id"] for ex in test], memory_contexts(train_log[:end], [ex["question"] for ex in test])))
        rows = evaluate_test(model, args, test, contexts, os.path.join(run_dir, f"test_ckpt{k}_{pct}pct.jsonl"))

        chunk = train_log[boundaries[k - 1]:end] if k > 0 else []
        summary.append({
            "checkpoint": k,
            "train_pct": pct,
            "n_train_seen": end,
            "test_accuracy": float(np.mean([r["correct"] for r in rows])),
            "test_correct": int(sum(r["correct"] for r in rows)),
            "n_test": len(rows),
            "test_no_answer": int(sum(r["prediction"] == NO_ANSWER for r in rows)),
            "train_chunk_online_accuracy": float(np.mean([r["correct"] for r in chunk])) if chunk else None,
            "cheatsheet_words": len(memory_text.split()),
            "memory_items": len(items) if items is not None else None,
            "mean_context_words": float(np.mean([len(c.split()) for c, _ in contexts.values()])),
        })
        json.dump(summary, open(os.path.join(run_dir, "summary.json"), "w"), indent=2)
        print(f"== Checkpoint {k} test accuracy: {summary[-1]['test_accuracy']:.3f} "
              f"({summary[-1]['test_correct']}/{summary[-1]['n_test']}, no answer: {summary[-1]['test_no_answer']})")

    print("\nTest accuracy vs. training data seen:")
    for s in summary:
        print(f"  {s['train_pct']:>3}% ({s['n_train_seen']:>3} ex): {s['test_accuracy']:.3f}")
    print(f"Results in {run_dir}")


if __name__ == "__main__":
    main(Arguments().parse_args())
