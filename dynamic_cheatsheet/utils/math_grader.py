"""
Verifiable-reward grading for math benchmarks (GSM8K, MATH500, AIME).

The functions are:
* extract_math_answer(output: str) -> str: Extracts the final answer from a generator output, falling back to the last \\boxed{...}.
* grade_math_answer(prediction: str, gold: str) -> bool: Checks a predicted answer against the gold answer.

Note: math_verify enforces timeouts with SIGALRM, so grade answers from the main thread.
"""

import re
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

from .extractor import extract_answer

NO_ANSWER = "No final answer found"


def _last_boxed(text: str) -> str:
    """Return the content of the last \\boxed{...} in the text (brace-balanced), or an empty string."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return ""
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j].strip()
    return ""


def extract_math_answer(output: str) -> str:
    """
    Extract the final answer from the generator output. Uses the <answer> block (DC format)
    and falls back to the last \\boxed{...} expression.
    """
    answer = extract_answer(output)
    if answer != NO_ANSWER and answer.strip():
        # A fenced answer (```latex ... ```) leaves its language tag on the first line
        return re.sub(r"^(latex|answer|plaintext|text|math|markdown)[ \t]*\n", "", answer.strip(), flags=re.IGNORECASE)
    # Common near-misses of the requested format: a ```answer code block, a bare line after
    # "FINAL ANSWER:", or a \boxed{...} answer
    fenced = re.findall(r"```answer\s*\n(.*?)```", output, flags=re.DOTALL)
    if fenced and fenced[-1].strip():
        return fenced[-1].strip()
    bare = _bare_final_answer(output)
    if bare:
        return bare
    boxed = _last_boxed(output)
    return boxed if boxed else NO_ANSWER


def _bare_final_answer(output: str) -> str:
    """First non-empty line after the last "FINAL ANSWER", without math delimiters or a \\text{} wrapper."""
    parts = re.split(r"FINAL ANSWER\s*:?", output)
    if len(parts) < 2:
        return ""
    boxed = _last_boxed(parts[-1])
    if boxed:
        return boxed
    lines = [l.strip() for l in parts[-1].splitlines() if l.strip()]
    if not lines:
        return ""
    line = lines[0]
    # A display-math block opened on its own line (\[ ... \] or $$ ... $$): take the whole block
    closing = {"\\[": "\\]", "$$": "$$"}.get(line)
    if closing:
        block = []
        for l in lines[1:]:
            if l.startswith(closing):
                break
            block.append(l)
        line = " ".join(block)
    line = line.replace("\\text{-}", "-")
    for pattern in (r"^\$+(.*?)\$+$", r"^\\\[(.*)\\\]$", r"^\\\((.*)\\\)$"):
        line = re.sub(pattern, r"\1", line).strip()
    # Unwrap \text{...} only when it encloses the whole answer (not "\text{-}\frac{\pi}{6}")
    if line.startswith("\\text{") and _last_boxed("\\boxed{" + line[len("\\text{"):]) == line[len("\\text{"):-1].strip():
        line = line[len("\\text{"):-1].strip()
    return line


def _normalize_number(text: str) -> str:
    """Strip common decorations from a numeric answer (e.g. '$1,234.00' -> '1234')."""
    text = text.replace("\\$", "").replace("\\%", "").replace("%", "")
    text = text.strip().strip("$").strip()
    text = re.sub(r"\\boxed\{(.*)\}", r"\1", text)
    text = text.replace(",", "").replace("\\!", "").replace(" ", "")
    text = re.sub(r"\\text\{[^}]*\}", "", text)
    try:
        value = float(text)
        return str(int(value)) if value == int(value) else str(value)
    except ValueError:
        return text


def _parse_math(text: str):
    # Without math delimiters, math_verify may only extract a fragment (e.g. "3" from "3\sqrt{3}")
    config = [LatexExtractionConfig(), ExprExtractionConfig()]
    if "$" not in text and "\\boxed" not in text:
        parsed = parse(f"${text}$", extraction_config=config)
        if parsed:
            return parsed
    return parse(text, extraction_config=config)


def grade_math_answer(prediction: str, gold: str) -> bool:
    """
    Return True if the predicted answer matches the gold answer.
    Tries exact numeric matching first, then symbolic equivalence via math_verify.
    """
    if prediction is None or prediction == NO_ANSWER or not prediction.strip():
        return False

    if _normalize_number(prediction) == _normalize_number(gold):
        return True

    try:
        gold_parsed = _parse_math(gold)
        pred_parsed = _parse_math(prediction)
        if not gold_parsed or not pred_parsed:
            return False
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return False
