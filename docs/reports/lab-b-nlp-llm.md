# Small models, measured: representations, classifiers, retrieval and RAG on a laptop CPU

*Lab B technical report · [ai-research-labs](../../README.md) · code in [`lab-b-nlp-llm/`](../../lab-b-nlp-llm/) ·
detailed per-module reports in [`lab-b-nlp-llm/reports/`](../../lab-b-nlp-llm/reports/)*

## Abstract

We compare the NLP building blocks a small team can run without a GPU — sparse vectors, static
and contextual embeddings, fine-tuned encoders, retrieval pipelines and 0.5B–1.5B-parameter
instruction-tuned LLMs — on text classification, search and retrieval-augmented generation
(RAG), always reporting accuracy together with its confidence interval and its cost. No
representation wins on every task (Friedman p = 0.54): bag-of-words beats sentence embeddings by
16 points on emotion detection. Prompting a 1.5B LLM is the best classifier for sentiment (94.0%)
but is beaten by 128 labelled examples on topic classification and costs 0.3–1.6 s per text. In
search, hybrid BM25 + dense retrieval with a cross-encoder reaches nDCG@10 0.929 on SQuAD but the
re-ranker does not transfer to scientific claims (SciFact). RAG lifts the answer F1 of
Qwen2.5-0.5B from 0.10 to 0.61, yet the system almost never abstains on unanswerable questions
and a single planted passage hijacks 60–95% of its answers. An LLM judge agrees with
reference-based labels on 90% of answers (κ = 0.79).

## 1. Questions

1. **Representations (B1).** Do sentence embeddings beat TF-IDF, and at what encoding cost?
2. **Classification (B2).** How many labelled examples does a classic model need to beat a
   zero- or few-shot LLM?
3. **Search (B3).** BM25 vs. dense vs. hybrid vs. re-ranking — and how large should chunks be?
4. **RAG (B4).** Does retrieval help a small LLM, does it abstain when the answer is not in the
   corpus, can it be manipulated through the corpus, and can an LLM judge evaluate it?
5. **Cost (B5).** Which configurations are on the quality-latency Pareto frontier?

## 2. Setup

| | |
|---|---|
| Data | AG News, SST-2, Emotion (classification); SQuAD v1.1 dev as an open-domain corpus (1,783 passages, 6 articles held out) and BEIR SciFact (5,183 abstracts) for search and RAG — see the [data cards](../data-cards.md) |
| Models | bag-of-words and TF-IDF; `potion-base-8M` (static embeddings); `all-MiniLM-L6-v2`, `bge-small-en-v1.5` (sentence transformers); `ms-marco-MiniLM-L-6-v2` (cross-encoder); DistilBERT (fine-tuned); `Qwen2.5-0.5B-Instruct` and `Qwen2.5-1.5B-Instruct` via 🤗 Transformers |
| LLM classification | the label is chosen by comparing label log-probabilities, so every prediction is valid; the few-shot prompt's KV cache is computed once and reused |
| Retrieval | BM25 on sparse matrices (validated: nDCG@10 0.665 on SciFact, the published BEIR value); FAISS inner-product index; reciprocal-rank fusion (k = 60); cross-encoder over the top 20 |
| RAG | 100 answerable questions + 40 about held-out articles; greedy decoding, ≤ 32 new tokens; generations cached on disk |
| Statistics | bootstrap 95% CIs; McNemar against the best classifier; Wilcoxon signed-rank on per-query nDCG@10 and per-answer F1 with Holm correction; Friedman across datasets; Cohen's κ for the judge |
| Compute | 4-core CPUs only (laptop; B2 on 4-vCPU GitHub-hosted runners, one parallel job per dataset); timings report the median per text |

## 3. Results

### 3.1 Representations (B1)

Logistic regression on each representation (5k training / 2k test texts):

| Representation | AG News | SST-2 | Emotion | Texts/s on CPU (AG News) |
|---|---|---|---|---|
| bag-of-words | 85.3% | 75.5% | **85.9%** | 50,021 |
| TF-IDF | 87.7% | 75.5% | 80.3% | 17,574 |
| potion-base-8M (static) | 87.6% | 77.4% | 69.2% | 20,536 |
| all-MiniLM-L6-v2 | 87.7% | 79.9% | 67.3% | 93 |
| bge-small-en-v1.5 | **89.9%** | **86.0%** | 70.0% | 46 |

Contextual embeddings win on topic and sentiment but lose by 16 points on Emotion, whose labels
hinge on specific words that sentence embeddings blur. Across the three datasets no
representation is better in general (Friedman p = 0.54). Static embeddings give TF-IDF-level
accuracy at TF-IDF-level speed in 256 dense dimensions.

### 3.2 Trained models vs. prompting (B2)

Accuracy at the largest training size (300 test texts per dataset, hence wide CIs):

| Method | Labels | AG News | SST-2 | Emotion | ms per text |
|---|---|---|---|---|---|
| TF-IDF + LR | 8,192 | 86.3% | 76.0% | 78.0% | 0.08 |
| embeddings + LR | 8,192 | 88.7% | 82.3% | 66.7% | ~25 |
| DistilBERT fine-tuned | ≤ 8,192 | **89.0%** | 80.0% | **86.7%** | ~40 |
| Qwen2.5-1.5B zero-shot | 0 | 73.7% | 90.3% | 49.7% | ~1,600 |
| Qwen2.5-1.5B few-shot | 1 per class | 72.3% | **94.0%** | 49.0% | ~1,600 |

On AG News, 128 labelled examples with frozen embeddings already beat every LLM setting; on SST-2
few-shot prompting beats every trained model up to 8,192 labels; on Emotion the LLMs stay below
50%. Prompting is the right tool only where the label is a concept the model already knows
(sentiment), not a dataset-specific taxonomy.

### 3.3 Search (B3)

| Retriever | SQuAD nDCG@10 | SciFact nDCG@10 | ms/query (SQuAD) |
|---|---|---|---|
| BM25 | 0.853 | 0.665 | < 1 |
| bge-small (dense) | 0.805 | 0.706 | 6.8 |
| hybrid (RRF) | 0.869 | **0.707** | 6.4 |
| hybrid + cross-encoder | **0.929** | 0.696 | 671 |

On SQuAD, questions reuse the passage's words, so BM25 beats dense retrieval and re-ranking adds
6 points at ~100× the latency. On SciFact dense and hybrid tie (Holm p = 0.76) and the MS MARCO
re-ranker does not help. Chunk size pulls the retrievers apart: BM25 finds the answer in its top
5 more often with larger chunks (87.8% → 95.6% from 64 to 512 words, at 8× the prompt length),
dense retrieval prefers small ones.

### 3.4 Retrieval-augmented generation (B4)

| Model | Setting | Token F1 (95% CI) | Exact match |
|---|---|---|---|
| Qwen2.5-0.5B | closed book | 0.10 [0.06, 0.15] | 5% |
| Qwen2.5-0.5B | RAG, top 1 | 0.61 [0.52, 0.69] | 53% |
| Qwen2.5-0.5B | RAG, top 3 | 0.53 [0.44, 0.62] | 46% |
| Qwen2.5-1.5B | closed book | 0.13 [0.08, 0.20] | 7% |
| Qwen2.5-1.5B | RAG, top 3 | 0.69 [0.61, 0.78] | 61% |
| Qwen2.5-1.5B | gold passage | 0.83 [0.76, 0.89] | 70% |

**Abstention.** On 40 questions whose article is not in the corpus, the 0.5B model never says
"I don't know" (95–100% hallucinated answers); the 1.5B model abstains 28% of the time when the
prompt allows it and 0% otherwise.

**Attacks through the corpus.** An "ignore all previous instructions" passage never worked
(0% success). A planted passage stating a false answer — corpus poisoning — hijacked 85–90% of
answers, and a defensive system prompt did not reliably help (60% for 0.5B, 95% for 1.5B).

**LLM as a judge.** Qwen2.5-1.5B grading 200 answers by comparing the probabilities of
*Yes*/*No* agrees with the reference-based label on 90% of them (κ = 0.79); its errors are mostly
lenient (17 wrong answers accepted vs. 2 right answers rejected).

### 3.5 Cost and latency (B5)

| Configuration | F1 | p50 latency | Time to first token | Decoding |
|---|---|---|---|---|
| **0.5B, RAG top 1** | 0.61 | 1.45 s | 1.13 s | 15.5 tok/s |
| 0.5B, RAG top 5 | 0.52 | 4.42 s | 4.11 s | 13.9 tok/s |
| 1.5B, RAG top 3 | 0.69 | 9.34 s | 8.69 s | 5.4 tok/s |

The 0.5B model with one passage is on the Pareto frontier; every extra passage adds prefill time
before the first token, and the 1.5B model decodes ~3× slower.

## 4. Discussion

- **Baselines first.** TF-IDF and BM25 were within a few points of the best methods on half the
  tasks at 100–10,000× lower cost; the strongest method changed with the dataset every time.
- **A small labelled set is often worth more than a bigger prompt.** The crossover point was
  measured, not assumed: 128 labels on topics, more than 8,192 on sentiment.
- **The retriever is part of the model.** With a good retriever (gold passage in the top 1 for
  86% of questions) a 0.5B model answers well; more context *hurt* it.
- **RAG's failure modes are about trust, not accuracy.** Small models do not abstain, and they
  believe whatever the corpus says. Prompt-level defences are not a security boundary: access
  control on what enters the index is.

## 5. Limitations

- Small evaluation sets for the expensive experiments (300 texts per dataset in B2, 100 + 40
  questions in B4); CIs are reported but are wide (± 4–9 points).
- English only; two LLM sizes from one family; greedy decoding only. Hosted models are supported
  by the `LLM` interface but were not run (no API budget), so B5 has no USD column filled.
- SQuAD as a RAG benchmark favours lexical retrieval and short extractive answers.
- The judge is validated against a reference-based label, not yet against human labels
  (`python -m lab_b.annotate` collects them).
- Latencies are CPU- and machine-specific; relative comparisons are more meaningful than
  absolute numbers.

## 6. Reproducibility

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,lab-b]"
python -m lab_b.representations --config lab-b-nlp-llm/configs/b1_representations.yaml
python -m lab_b.search          --config lab-b-nlp-llm/configs/b3_search.yaml
python -m lab_b.rag             --config lab-b-nlp-llm/configs/b4_rag.yaml
gh workflow run experiments.yml -f experiment=b2_classification   # B2 on GitHub Actions
```

## References

- Robertson, Zaragoza (2009). *The probabilistic relevance framework: BM25 and beyond.*
- Reimers, Gurevych (2019). *Sentence-BERT.* EMNLP · Xiao et al. (2023). *C-Pack: packaged resources to advance general Chinese embedding* (BGE).
- Cormack, Clarke, Büttcher (2009). *Reciprocal rank fusion.* SIGIR · Nogueira, Cho (2019). *Passage re-ranking with BERT.* arXiv.
- Thakur et al. (2021). *BEIR: a heterogeneous benchmark for zero-shot evaluation of IR models.* NeurIPS Datasets & Benchmarks.
- Lewis et al. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks.* NeurIPS.
- Sanh, Debut, Chaumond, Wolf (2019). *DistilBERT.* arXiv · Qwen Team (2024). *Qwen2.5 technical report.* arXiv.
- Zheng et al. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.* NeurIPS · Cohen (1960). *A coefficient of agreement for nominal scales.*
- Greshake et al. (2023). *Not what you've signed up for: indirect prompt injection.* AISec · Zou et al. (2025). *PoisonedRAG.* USENIX Security.
- Rajpurkar et al. (2016). *SQuAD.* EMNLP · Wadden et al. (2020). *Fact or fiction: verifying scientific claims* (SciFact). EMNLP.
- Zhang, Zhao, LeCun (2015). *Character-level convolutional networks for text classification* (AG News). NeurIPS · Socher et al. (2013). *Recursive deep models for semantic compositionality* (SST). EMNLP · Saravia et al. (2018). *CARER: contextualized affect representations for emotion recognition.* EMNLP.
