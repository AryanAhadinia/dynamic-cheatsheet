"""
Append-only memory with embedding retrieval.

Instead of a single cheatsheet that the curator rewrites after every problem, the curator writes a few new
memory items per problem, which are appended to a library. For each new question, the top-k most similar
items are retrieved and shown to the generator. The memory at any checkpoint is therefore a superset of the
memory at every earlier checkpoint, while the generator's context stays a fixed size.

The functions/classes are:
* parse_memory_items(text: str) -> List[str]: Extracts <memory_item> blocks from a curator response.
* Embedder: Batched, cached embeddings from an OpenAI-compatible embeddings endpoint.
* AppendMemory: Retrieves the top-k memory items for a batch of questions and formats them as a cheatsheet.
"""

import os
import random
import re
import time
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI

EMPTY_MEMORY = "(empty)"

# Qwen3-Embedding expects an instruction on the query side only
QUERY_INSTRUCTION = (
    "Instruct: Given a math problem, retrieve notes describing techniques, formulas, and pitfalls that help solve it\n"
    "Query: "
)

MEMORY_HEADER = (
    "The memory items below were written after solving past problems and retrieved as possibly relevant to the "
    "current question (ordered from least to most similar). Use them when they apply, but verify before relying on them.\n\n"
)


def parse_memory_items(text: str) -> List[str]:
    """Return the <memory_item>...</memory_item> blocks in the text (tags included), in order."""
    return [m.strip() for m in re.findall(r"<memory_item>.*?</memory_item>", text, flags=re.DOTALL)]


class Embedder:
    def __init__(self, model: str, base_url: str = "https://api.swissai.svc.cscs.ch/v1",
                 api_key_env: str = "SWISSAI_API_KEY", batch_size: int = 64, max_chars: int = 8000) -> None:
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=os.environ[api_key_env])
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.cache: Dict[str, np.ndarray] = {}

    def _request(self, texts: List[str], retries: int = 20) -> List[List[float]]:
        for attempt in range(retries):
            try:
                response = self.client.embeddings.create(model=self.model, input=texts)
                return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
            except Exception as e:
                if attempt == retries - 1:
                    raise
                # The endpoint allows ~15 embedding requests/minute, shared by all parallel runs
                wait = 60 + random.uniform(0, 30) if "429" in str(e) else 30 * min(attempt + 1, 4)
                print(f"  ! Embedding request failed ({str(e)[:120]}); retrying in {wait:.0f}s")
                time.sleep(wait)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized embeddings (len(texts), dim); repeated texts are served from the cache."""
        texts = [t[:self.max_chars] for t in texts]
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            for text, vector in zip(batch, self._request(batch)):
                vector = np.asarray(vector, dtype=np.float32)
                self.cache[text] = vector / (np.linalg.norm(vector) + 1e-12)
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 0), dtype=np.float32)


class AppendMemory:
    def __init__(self, embedder: Embedder, top_k: int) -> None:
        """top_k = -1 shows every memory item (no retrieval)."""
        self.embedder = embedder
        self.top_k = top_k

    @staticmethod
    def _document(item: dict) -> str:
        # Index each item together with the problem it came from: new questions mostly resemble past problems
        return f"{item['text']}\n\nSource problem: {item['source_question']}"

    def contexts(self, items: List[dict], questions: List[str]) -> List[Tuple[str, List[str]]]:
        """For each question, return (formatted memory context, ids of the retrieved items)."""
        if not items:
            return [(EMPTY_MEMORY, []) for _ in questions]
        if self.top_k < 0:
            # Whole memory in insertion order (only viable while it fits in the context window)
            selections = [list(range(len(items)))] * len(questions)
        else:
            # Most similar item last, closest to the question
            selections = [order[:self.top_k][::-1] for order in self._rank(items, questions)]
        results = []
        for chosen in selections:
            body = "\n\n".join(items[i]["text"] for i in chosen)
            results.append((MEMORY_HEADER + body, [items[i]["id"] for i in chosen]))
        return results

    def embed_queries(self, questions: List[str]) -> None:
        """Pre-compute (and cache) query embeddings in bulk."""
        self.embedder.embed([QUERY_INSTRUCTION + q for q in questions])

    def _rank(self, items: List[dict], questions: List[str]) -> List[List[int]]:
        item_vectors = self.embedder.embed([self._document(item) for item in items])
        query_vectors = self.embedder.embed([QUERY_INSTRUCTION + q for q in questions])
        similarities = query_vectors @ item_vectors.T
        return [list(np.argsort(-row, kind="stable")) for row in similarities]
