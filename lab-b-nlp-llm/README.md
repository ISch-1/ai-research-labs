# Lab B — NLP & Large Language Models

> From bag-of-words to retrieval-augmented generation, measured end to end on a laptop CPU:
> which text representation, which classifier, which retriever and which LLM configuration —
> and what each one costs in latency and tokens. Every comparison comes with a significance
> test; every LLM claim is checked against references, an LLM judge and (optionally) human labels.

## Key findings

**B1 — Representations: contextual embeddings usually win, but not on every task.**
- `bge-small` sentence embeddings are the most accurate on AG News (89.9%) and SST-2 (86.0%),
  while encoding two to three orders of magnitude fewer texts per second than TF-IDF on CPU.
- On Emotion, plain bag-of-words (85.9%) beats the best embedding (bge-small, 70.0%) by 16 points: the labels hinge on
  specific emotion words that sentence embeddings blur. Across the three datasets no
  representation is better in general (Friedman p = 0.54).

**B2 — Trained models vs. prompting: it depends on the task.**
- Topic classification (AG News): with only 128 labelled examples a frozen embedding + logistic
  regression already beats zero-shot Qwen2.5-1.5B (73.7%); fine-tuned DistilBERT reaches 89.0%.
- Sentiment (SST-2): few-shot Qwen2.5-1.5B (94.0%) beats every trained model up to 8,192 labels
  (best: embeddings, 82.3%) — the one task where prompting clearly wins.
- Emotion: the LLMs stay below 50%; fine-tuning reaches 86.7%.
- Cost: an LLM label costs 0.3–1.6 s on 4 CPU cores vs. ~0.1 ms for TF-IDF and 7–40 ms for
  embeddings or DistilBERT.

**B3 — Search: hybrid + re-ranking on Wikipedia, dense on science.**
- On SQuAD a cross-encoder over hybrid retrieval reaches nDCG@10 0.929 (vs. 0.853 for BM25), but
  costs ~670 ms per query instead of under 10 ms. Dense retrieval alone (0.805) is *worse* than
  BM25 here: SQuAD questions reuse the paragraph's words.
- On SciFact dense and hybrid retrieval tie (0.706–0.707, Holm p = 0.76) and re-ranking does not
  help (0.696): the MS MARCO cross-encoder does not transfer to scientific claims.
- Chunk size pulls the two retrievers apart: BM25 finds more answers with larger chunks (hit@5
  0.878 → 0.956 from 64 to 512 words, at 8× the prompt length), dense retrieval prefers small ones.

**B4 — RAG: large gains, weak abstention, and poisoning works.**
- Retrieval lifts token F1 from 0.10 to 0.61 (Qwen2.5-0.5B, top passage) and from 0.13 to 0.69
  (1.5B, top 3); with the gold passage the 1.5B model reaches 0.83.
- More context hurts the small model: 0.61 with one passage, 0.53 with three.
- Asked about articles that are not in the corpus, the 0.5B model never says "I don't know"; the
  1.5B model abstains 28% of the time when told it may, 0% when not told.
- An "ignore previous instructions" passage never worked, but a planted passage stating a false
  answer hijacked 85–90% of answers, and the defensive prompt barely helped (60–95%).
- The LLM judge agrees with the reference-based label on 90% of answers (Cohen's κ = 0.79).

**B5 — Cost and latency.** The 0.5B model with one retrieved passage (F1 0.61, 1.5 s per answer)
dominates the Pareto frontier for latency; the 1.5B model is more accurate but ~3× slower per
token (≈5 vs. ≈15 tokens/s), and every extra passage adds prefill time before the first token.


## Modules

| Module | Question | Code | Report |
|---|---|---|---|
| B1 | Does a sentence transformer beat TF-IDF — and at what encoding cost? | [`representations.py`](src/lab_b/representations.py) | [B1](reports/b1_representations.md) |
| B2 | How many labelled examples does a classic model need to beat a zero/few-shot LLM? | [`classification.py`](src/lab_b/classification.py) | [B2](reports/b2_classification.md) |
| B3 | BM25 vs. dense vs. hybrid vs. re-ranking; how big should chunks be? | [`search.py`](src/lab_b/search.py), [`retrieval.py`](src/lab_b/retrieval.py) | [B3](reports/b3_search.md) |
| B4 | Does RAG help a small LLM, does it abstain when it should, can it be prompt-injected, and can an LLM judge be trusted? | [`rag.py`](src/lab_b/rag.py) | [B4–B5](reports/b4_rag.md) |
| B5 | Quality vs. latency vs. tokens: model size, number of passages, Pareto frontier | [`rag.py`](src/lab_b/rag.py) | [B4–B5](reports/b4_rag.md#b5--cost-latency-and-quality) |

## Methodology

**Models.** Everything runs locally on CPU: sparse baselines (scikit-learn), static embeddings
(`potion-base-8M`, model2vec), sentence transformers (`all-MiniLM-L6-v2`, `bge-small-en-v1.5`),
a cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`), DistilBERT for fine-tuning and two
instruction-tuned LLMs (`Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct`) via 🤗 Transformers.
The `LLM` interface ([`llm.py`](src/lab_b/llm.py)) is backend-agnostic, so a hosted model can be
added without touching the experiments.

**B1.** The same classifier (logistic regression, C by 3-fold CV) on each representation of AG
News, SST-2 and Emotion (5k train / 2k test); McNemar against the best; throughput measured
without the embedding cache.

**B2.** Nested training subsets (32 → 8,192 examples) give learning curves for TF-IDF, frozen
embeddings and fine-tuned DistilBERT. The LLMs classify by comparing the log-probability of each
label — no generation, no answer parsing, every prediction valid — zero-shot and with one example
per class; the few-shot prompt's KV cache is computed once and reused for every test text
(tested to give the same scores as the full prompt).

**B3.** SQuAD v1.1 dev turned into an open-domain corpus (every paragraph a passage, 1,000
questions) and BEIR SciFact (5k abstracts, graded relevance). BM25 is implemented with sparse
matrices and validated against the BEIR reference (nDCG@10 0.665 on SciFact, identical to the
published value). Wilcoxon signed-rank on per-query nDCG@10 with Holm correction. Chunking: SQuAD
articles re-split into 64–512-word windows; recall vs. the number of context words a prompt pays.

**B4–B5.** 100 SQuAD questions under closed-book, oracle-passage and RAG (top 1/3/5) settings;
EM, token F1, a lenient "contains the answer" score and groundedness. Six whole articles are held
out of the corpus to create questions RAG *cannot* answer (abstention vs. hallucination vs.
answering from memory). Prompt injection: one retrieved passage is replaced by an instruction,
with and without a defensive prompt. An LLM judge is scored against the reference-based label
(Cohen's κ) and, after `python -m lab_b.annotate`, against human labels. B5 reports p50/p95
latency, time-to-first-token, prompt/completion tokens and the quality-latency Pareto frontier.
All generations are cached on disk, so runs resume after interruption.

## Run

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build (skip with a GPU)
pip install -e ".[dev,lab-b]"

python -m lab_b.search          --config lab-b-nlp-llm/configs/b3_search.yaml
python -m lab_b.rag             --config lab-b-nlp-llm/configs/b4_rag.yaml        # ~2 h on CPU
python -m lab_b.representations --config lab-b-nlp-llm/configs/b1_representations.yaml
python -m lab_b.classification  --config lab-b-nlp-llm/configs/b2_classification.yaml
python -m lab_b.annotate --answers lab-b-nlp-llm/reports/results/b4_rag_answers.csv   # optional
```

Models (~4.5 GB) download from the Hugging Face Hub on first use; datasets are cached in `./data`.

**On GitHub Actions instead of a laptop.** The `Experiments` workflow runs a long experiment on
GitHub-hosted runners, one parallel job per dataset; each job caches its predictions and timings,
and a final job rebuilds the full report from those caches without loading any model:

```bash
gh workflow run experiments.yml --ref main -f experiment=b2_classification
```

Reports state the hardware their timings were measured on.
Every config has an offline `*_smoke.yaml` twin (synthetic data and a deterministic fake LLM)
that CI runs; `tests/lab_b/test_hf_models.py` exercises the real model code with a 135M model.
