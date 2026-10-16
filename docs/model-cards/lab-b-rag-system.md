# System card — Lab B retrieval-augmented QA pipeline

*Format after Mitchell et al. (2019), adapted to a system of off-the-shelf models. Lab B trains no
generative model; this card describes the pipeline evaluated in B3–B5 and what was measured about
it. Report: [Lab B](../reports/lab-b-nlp-llm.md).*

## System details

| Component | Model / method | License (upstream) |
|---|---|---|
| Sparse retrieval | Okapi BM25 (k₁ = 1.5, b = 0.75), implemented with sparse matrices | — (code MIT) |
| Dense retrieval | `BAAI/bge-small-en-v1.5`, FAISS inner-product index | MIT |
| Fusion | reciprocal-rank fusion, k = 60 | — |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the top 20 | Apache-2.0 |
| Generation | `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen2.5-1.5B-Instruct`, greedy, ≤ 32 new tokens | Apache-2.0 |
| Judge | `Qwen2.5-1.5B-Instruct`, *Yes*/*No* log-probability comparison | Apache-2.0 |

All models run locally on CPU through 🤗 Transformers; no data leaves the machine. The `LLM`
interface ([`llm.py`](../../lab-b-nlp-llm/src/lab_b/llm.py)) allows a hosted model to be
swapped in.

## Intended use

- **Intended:** research and teaching — measuring how retrieval, model size and context length
  trade off quality and latency; a reference implementation of RAG evaluation.
- **Out of scope:** answering questions for end users in production, any medical, legal or
  financial advice, and any corpus that untrusted parties can write to (see the poisoning result).

## Evaluation data and metrics

SQuAD v1.1 dev as an open-domain corpus (42 articles, 1,783 passages) with 100 answerable
questions and 40 questions about 6 held-out articles; BEIR SciFact for retrieval only. Metrics:
nDCG@10 / Recall@k (retrieval), exact match and token F1 with bootstrap CIs (answers), abstention
and hallucination rates, attack success rate, latency p50/p95, time to first token, Cohen's κ.

## Quantitative analysis

| Measure | 0.5B | 1.5B |
|---|---|---|
| Token F1, closed book | 0.10 | 0.13 |
| Token F1, RAG (best k) | 0.61 (top 1) | 0.69 (top 3) |
| Token F1, gold passage | 0.68 | 0.83 |
| Abstains on unanswerable questions (instructed) | 0% | 28% |
| Hallucinates on unanswerable questions (instructed) | 95% | 62% |
| Prompt-injection success ("ignore previous instructions") | 0% | 0% |
| Corpus-poisoning success, plain / defended prompt | 85% / 60% | 90% / 95% |
| Median latency per answer (RAG setting above, 4-core CPU) | 1.45 s | 9.34 s |

Retriever (hybrid + re-ranking) on SQuAD: nDCG@10 0.929, gold passage in the top 1 / 3 / 5 for
86% / 92% / 95% of questions. The judge agrees with the reference-based label on 90% of answers
(κ = 0.79) and errs on the lenient side.

## Ethical considerations and risks

- **Hallucination.** Small models almost never decline; an answer is not evidence that the corpus
  contains it. Show the retrieved passages with every answer.
- **Corpus poisoning.** One planted passage that ranks first controls most answers, and a
  defensive system prompt does not prevent it. Treat write access to the index as the security
  boundary.
- **Judge bias.** An LLM judge from the same family as the generator may share its blind spots;
  it has not yet been validated against human labels (`python -m lab_b.annotate`).

## Caveats

English Wikipedia and scientific abstracts only; short extractive answers; two model sizes from
one family; greedy decoding. Latencies are specific to the CPUs they were measured on.
