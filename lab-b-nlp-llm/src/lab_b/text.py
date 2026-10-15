"""Answer normalization and the official SQuAD metrics (exact match, token F1)."""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)

ABSTAIN = "I don't know"
_ABSTAIN_PATTERNS = re.compile(
    r"\b(i (do not|don't|dont) know|not (mentioned|provided|stated|specified|available)|"
    r"(cannot|can't|unable to) (answer|determine|find)|no (information|answer))\b",
    re.IGNORECASE,
)


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, articles and extra whitespace (SQuAD v1.1 script)."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(a) for a in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    """Best bag-of-tokens F1 over the reference answers."""
    pred = normalize_answer(prediction).split()
    best = 0.0
    for answer in answers:
        gold = normalize_answer(answer).split()
        common = Counter(pred) & Counter(gold)
        overlap = sum(common.values())
        if not pred or not gold:
            best = max(best, float(pred == gold))
        elif overlap:
            precision, recall = overlap / len(pred), overlap / len(gold)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_answer(prediction: str, answers: list[str]) -> float:
    """Lenient correctness: a reference answer appears inside a (possibly verbose) prediction."""
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(a) and normalize_answer(a) in pred for a in answers))


def is_abstention(prediction: str) -> bool:
    return bool(_ABSTAIN_PATTERNS.search(prediction)) or not prediction.strip()


def words(text: str) -> int:
    return len(text.split())
