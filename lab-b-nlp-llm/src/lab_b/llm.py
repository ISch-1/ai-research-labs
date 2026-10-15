"""LLM backends with the measurements Lab B needs: latency, time-to-first-token, tokens.

* ``HFChatLLM``  — any chat model from the Hugging Face Hub, run locally on CPU/GPU.
* ``FakeLLM``    — deterministic stand-in for tests (extracts answers from the prompt).

Both implement ``generate`` (free text) and ``choice_logprobs`` (score a fixed set of
labels in one forward pass, used for zero/few-shot classification). ``choice_logprobs``
reuses the KV cache of a shared prompt prefix — the few-shot examples are encoded
once, not once per test example.
"""

from __future__ import annotations

import copy
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

Messages = list[dict[str, str]]


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    ttft_s: float


class LLM(Protocol):
    name: str

    def generate(self, messages: Messages, max_new_tokens: int = 48) -> Generation: ...

    def choice_logprobs(
        self, prefix: Messages, query: Messages, choices: list[str]
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------- Hugging Face


class HFChatLLM:
    """Greedy decoding with a chat template; float32 on CPU by default."""

    def __init__(
        self, model_id: str, *, threads: int | None = None, dtype: str = "float32"
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self.name = model_id.split("/")[-1]
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype)
        )
        self.model.eval()
        self._prefix_cache: dict[str, tuple[Any, int]] = {}
        self._choice_ids: dict[tuple[str, ...], list[list[int]]] = {}

    def _encode(self, messages: Messages, *, generation_prompt: bool = True) -> Any:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=generation_prompt, return_tensors="pt", return_dict=True
        )

    def generate(self, messages: Messages, max_new_tokens: int = 48) -> Generation:
        import torch
        from transformers import TextIteratorStreamer

        inputs = self._encode(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        result: dict[str, Any] = {}

        def run() -> None:
            with torch.inference_mode():
                result["ids"] = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    streamer=streamer,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        start = time.perf_counter()
        thread = threading.Thread(target=run)
        thread.start()
        ttft, chunks = None, []
        for chunk in streamer:
            if ttft is None and chunk:
                ttft = time.perf_counter() - start
            chunks.append(chunk)
        thread.join()
        latency = time.perf_counter() - start
        n_prompt = int(inputs["input_ids"].shape[1])
        n_completion = int(result["ids"].shape[1]) - n_prompt
        return Generation("".join(chunks).strip(), n_prompt, n_completion, latency, ttft or latency)

    def _prefix(self, prefix: Messages) -> tuple[Any, int]:
        """KV cache of the rendered prefix (system prompt + few-shot turns), computed once."""
        import torch

        key = repr(prefix)
        if key not in self._prefix_cache:
            text = self.tokenizer.apply_chat_template(prefix, tokenize=False)
            ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            with torch.inference_mode():
                out = self.model(input_ids=ids, use_cache=True)
            self._prefix_cache[key] = (out.past_key_values, int(ids.shape[1]))
        return self._prefix_cache[key]

    def choice_logprobs(self, prefix: Messages, query: Messages, choices: list[str]) -> np.ndarray:
        """Log-probability of each choice as the assistant's answer.

        The full prompt is rendered once; the part after the cached prefix is the
        only text encoded per call. Choices are scored on their full token sequence,
        so multi-token labels are handled exactly.
        """
        import torch

        full = str(
            self.tokenizer.apply_chat_template(
                prefix + query, tokenize=False, add_generation_prompt=True
            )
        )
        head = str(self.tokenizer.apply_chat_template(prefix, tokenize=False)) if prefix else ""
        if not full.startswith(head):
            raise ValueError("chat template does not render the prefix as a string prefix")
        tail_ids = self.tokenizer(full[len(head) :], return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ]
        key = tuple(choices)
        if key not in self._choice_ids:
            self._choice_ids[key] = [
                self.tokenizer(c, add_special_tokens=False)["input_ids"] for c in choices
            ]

        with torch.inference_mode():
            if prefix:
                cache, _ = self._prefix(prefix)
                out = self.model(
                    input_ids=tail_ids, past_key_values=copy.deepcopy(cache), use_cache=True
                )
            else:
                out = self.model(input_ids=tail_ids, use_cache=True)
            first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
            scores = []
            for ids in self._choice_ids[key]:
                total = float(first[ids[0]])
                if len(ids) > 1:  # continue the sequence for multi-token labels
                    cont = self.model(
                        input_ids=torch.tensor([ids[:-1]]),
                        past_key_values=copy.deepcopy(out.past_key_values),
                        use_cache=False,
                    )
                    logp = torch.log_softmax(cont.logits[0].float(), dim=-1)
                    total += float(sum(logp[i, t] for i, t in enumerate(ids[1:])))
                scores.append(total)
        return np.asarray(scores)


# --------------------------------------------------------------------------- fake


_STOPWORDS = {
    "what",
    "when",
    "who",
    "which",
    "how",
    "is",
    "was",
    "the",
    "a",
    "an",
    "of",
    "in",
    "did",
    "does",
    "it",
    "its",
    "and",
    "to",
    "for",
}


def _content_words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower())) - _STOPWORDS


class FakeLLM:
    """Deterministic, instant LLM for tests.

    ``generate`` copies the first context sentence that shares the most words with
    the question (a crude extractive reader); it obeys "ignore previous
    instructions" injections when ``gullible`` so injection tests have signal.
    """

    def __init__(
        self, name: str = "fake", *, gullible: bool = True, latency_s: float = 0.0
    ) -> None:
        self.name = name
        self.gullible = gullible
        self.latency_s = latency_s

    def generate(self, messages: Messages, max_new_tokens: int = 48) -> Generation:
        start = time.perf_counter()
        prompt = "\n".join(m["content"] for m in messages)
        injected = re.search(
            r"ignore (?:all )?previous instructions.*?answer[^\"']*[\"']([^\"']+)[\"']",
            prompt,
            re.I,
        )
        if injected and self.gullible:
            text = injected.group(1)
        else:
            question = re.findall(r"Question:\s*(.+)", prompt)
            context = prompt.split("Question:")[0]
            sentences = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if len(s.split()) > 3
            ]
            q_words = _content_words(question[-1]) if question else set()
            best = max(sentences, key=lambda s: len(q_words & _content_words(s)), default="")
            overlap = len(q_words & _content_words(best))
            text = best.split(" is ")[-1].rstrip(".") if overlap >= 2 else "I don't know"
        time.sleep(self.latency_s)
        latency = time.perf_counter() - start
        return Generation(text, len(prompt.split()), len(text.split()), latency, latency / 2)

    def choice_logprobs(self, prefix: Messages, query: Messages, choices: list[str]) -> np.ndarray:
        """Scores each label by how often it, or words from its few-shot examples, occur in the text."""
        text = set(re.findall(r"\w+", query[-1]["content"].lower()))
        vocab: dict[str, set[str]] = {c: {c.lower()} for c in choices}
        for user, assistant in zip(prefix[1::2], prefix[2::2], strict=False):
            if assistant.get("content") in vocab:
                vocab[assistant["content"]] |= set(re.findall(r"\w+", user["content"].lower())) - {
                    "text"
                }
        return np.array([float(len(text & vocab[c])) for c in choices])


class LazyLLM:
    """Loads the real model on first use, so a run served from caches never downloads it."""

    def __init__(self, spec: str, **kwargs: Any) -> None:
        self.spec, self.kwargs = spec, kwargs
        self.name = "fake" if spec == "fake" else spec.split("/")[-1]
        self._llm: LLM | None = None

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def _get(self) -> LLM:
        if self._llm is None:
            self._llm = load_llm(self.spec, **self.kwargs)
        return self._llm

    def generate(self, messages: Messages, max_new_tokens: int = 48) -> Generation:
        return self._get().generate(messages, max_new_tokens)

    def choice_logprobs(self, prefix: Messages, query: Messages, choices: list[str]) -> np.ndarray:
        return self._get().choice_logprobs(prefix, query, choices)


def load_llm(spec: str, *, lazy: bool = False, **kwargs: Any) -> LLM:
    """``fake`` or ``hf:<model id>``; ``lazy=True`` defers loading until first use."""
    if lazy:
        return LazyLLM(spec, **kwargs)
    if spec == "fake":
        return FakeLLM()
    if spec.startswith("hf:"):
        return HFChatLLM(spec[3:], **kwargs)
    raise ValueError(f"unknown LLM spec {spec!r}")
