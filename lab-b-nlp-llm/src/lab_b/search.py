"""B3 — Semantic search: lexical vs. dense vs. hybrid retrieval, re-ranking and chunking.

    python -m lab_b.search --config lab-b-nlp-llm/configs/b3_search.yaml

1. Retrieval quality (nDCG@10, MRR@10, Recall@k) and cost (index time, ms/query)
   of each retriever on SQuAD (one relevant paragraph per question, Wikipedia)
   and SciFact (scientific claims, graded relevance). Per-query nDCG of the best
   retriever vs. each other: Wilcoxon signed-rank with Holm correction.
2. Chunking: SQuAD articles are re-split into fixed-size word windows; a chunk is
   relevant when it comes from the gold article and contains a gold answer. Larger
   chunks are found more often but cost more prompt tokens downstream (B4/B5).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field
from scipy.stats import wilcoxon

from airlab.config import ExperimentConfig
from airlab.experiment import Outputs, cli
from airlab.plots import plt, save
from airlab.reporting import fmt_p, md_table, write_results
from airlab.seed import set_seed
from airlab.stats import holm
from airlab.tracking import track_run
from lab_b import retrieval as R
from lab_b.datasets import QACorpus, load_qa
from lab_b.text import normalize_answer

DEFAULT_OUTPUT_DIR = Path("lab-b-nlp-llm/reports")


class SearchConfig(ExperimentConfig):
    output_dir: Path = DEFAULT_OUTPUT_DIR
    datasets: list[str]
    retrievers: list[str]
    n_questions: int | None = 1000
    chunk_dataset: str | None = "squad"
    chunk_sizes: list[int] = Field(default_factory=lambda: [64, 128, 256, 512])
    chunk_overlap: float = 0.25
    chunk_retrievers: list[str] = Field(default_factory=lambda: ["bm25"])
    # False: re-encode the corpus so "Index s" is the true cost (slower re-runs).
    cache_embeddings: bool = False


def relevance_by_index(qa: QACorpus) -> list[dict[int, int]]:
    pos = {p.id: i for i, p in enumerate(qa.passages)}
    return [{pos[d]: r for d, r in q.relevant.items() if d in pos} for q in qa.questions]


def evaluate(qa: QACorpus, spec: str, cache: bool) -> dict[str, Any]:
    start = time.perf_counter()
    retriever = R.build(spec, cache=cache).fit(qa.texts)
    index_s = time.perf_counter() - start
    queries = [q.text for q in qa.questions]
    retriever.search(queries[:2], 10)  # warm-up
    idx, per_query_s = R.timed_search(retriever, queries, 10)
    return {
        "metrics": R.ir_metrics(idx, relevance_by_index(qa)),
        "index_s": index_s,
        "ms": 1000 * per_query_s,
    }


def run_retrieval(qa: QACorpus, cfg: SearchConfig) -> tuple[list[str], list]:
    results = {spec: evaluate(qa, spec, cfg.cache_embeddings) for spec in cfg.retrievers}
    ranking = sorted(cfg.retrievers, key=lambda s: -results[s]["metrics"]["ndcg@10"].mean())
    best = ranking[0]
    raw = {}
    for s in ranking[1:]:
        a, b = results[best]["metrics"]["ndcg@10"], results[s]["metrics"]["ndcg@10"]
        raw[s] = 1.0 if np.allclose(a, b) else float(wilcoxon(a, b).pvalue)
    adj = holm(raw) if raw else {}
    rows = [
        {
            "experiment": cfg.experiment,
            "dataset": qa.name,
            "model": R.short_name(s),
            "fold": i,
            "metric": m,
            "value": float(v),
        }
        for s in cfg.retrievers
        for m in ("ndcg@10", "mrr@10", "recall@10")
        for i, v in enumerate(results[s]["metrics"][m])
    ]
    lines = [
        f"## `{qa.name}` — retrieval",
        "",
        f"{qa.source} · {len(qa.passages):,} passages · {len(qa.questions):,} queries",
        "",
        md_table(
            [
                "Retriever",
                "nDCG@10",
                "MRR@10",
                "Recall@1",
                "Recall@5",
                "Recall@10",
                "Index s",
                "ms/query",
                "Holm p vs best",
            ],
            [
                [
                    f"**{R.short_name(s)}**" if s == best else R.short_name(s),
                    *(
                        f"{results[s]['metrics'][m].mean():.3f}"
                        for m in ("ndcg@10", "mrr@10", "recall@1", "recall@5", "recall@10")
                    ),
                    f"{results[s]['index_s']:.1f}",
                    f"{results[s]['ms']:.1f}",
                    "—" if s == best else fmt_p(adj[s]),
                ]
                for s in ranking
            ],
        ),
        "",
    ]
    return lines, rows


def chunk_corpus(qa: QACorpus, size: int, overlap: float) -> tuple[list[str], list[str]]:
    """Concatenate each article's paragraphs and cut into word windows; returns (texts, article)."""
    by_article: dict[str, list[str]] = {}
    for p in qa.passages:
        by_article.setdefault(p.title, []).append(p.text)
    texts, owners = [], []
    step = max(1, int(size * (1 - overlap)))
    for title, paragraphs in by_article.items():
        words = " ".join(paragraphs).split()
        for start in range(0, max(1, len(words) - size + step), step):
            texts.append(f"{title}. " + " ".join(words[start : start + size]))
            owners.append(title)
    return texts, owners


def run_chunking(qa: QACorpus, cfg: SearchConfig, out: Outputs) -> tuple[list[str], list]:
    rows: list[dict[str, Any]] = []
    table: list[list[str]] = []
    points: dict[str, list[tuple[str, float, float]]] = {}
    paragraph = [(p.title, f"{p.title}. {p.text}") for p in qa.passages]
    variants: list[tuple[str, list[str], list[str]]] = [
        ("paragraph", [t for _, t in paragraph], [o for o, _ in paragraph])
    ] + [(f"{n} words", *chunk_corpus(qa, n, cfg.chunk_overlap)) for n in cfg.chunk_sizes]
    queries = [q.text for q in qa.questions]
    for spec in cfg.chunk_retrievers:
        for label, texts, owners in variants:
            norm = [normalize_answer(t) for t in texts]
            relevant = []
            for q in qa.questions:
                answers = [normalize_answer(a) for a in q.answers if normalize_answer(a)]
                title = q.title.replace("_", " ")
                relevant.append(
                    {
                        i: 1
                        for i, (o, t) in enumerate(zip(owners, norm, strict=True))
                        if o == title and any(a in t for a in answers)
                    }
                )
            idx, _ = R.build(spec).fit(texts).search(queries, 5)
            hits = {
                k: np.mean(
                    [
                        any(int(d) in rel for d in row[:k])
                        for row, rel in zip(idx, relevant, strict=True)
                    ]
                )
                for k in (1, 3, 5)
            }
            words5 = np.mean([sum(len(texts[int(d)].split()) for d in row[:5]) for row in idx])
            table.append(
                [
                    R.short_name(spec),
                    label,
                    f"{len(texts):,}",
                    *(f"{hits[k]:.3f}" for k in (1, 3, 5)),
                    f"{words5:,.0f}",
                ]
            )
            points.setdefault(spec, []).append((label, words5, hits[5]))
            for k in (1, 3, 5):
                rows.append(
                    {
                        "experiment": cfg.experiment,
                        "dataset": f"{qa.name}-chunking",
                        "model": f"{R.short_name(spec)} / {label}",
                        "fold": "all",
                        "metric": f"hit@{k}",
                        "value": float(hits[k]),
                    }
                )

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for spec, pts in points.items():
        ax.plot([p[1] for p in pts], [p[2] for p in pts], marker="o", label=R.short_name(spec))
        for label, x, y in pts:
            ax.annotate(label, (x, y), fontsize=7, xytext=(3, -9), textcoords="offset points")
    ax.set(
        xscale="log",
        xlabel="words in the top-5 chunks (≈ prompt cost)",
        ylabel="answer found in top 5",
        title="Chunk size: recall vs. context length",
    )
    ax.legend(fontsize=8)
    path, link = out.figure(f"{qa.name}_chunking.png")
    save(fig, path)
    lines = [
        f"## `{qa.name}` — chunk size",
        "",
        f"Articles re-split into windows with {cfg.chunk_overlap:.0%} overlap. Hit@k: a chunk "
        "from the right article containing a gold answer is in the top k.",
        "",
        md_table(
            ["Retriever", "Chunking", "Chunks", "Hit@1", "Hit@3", "Hit@5", "Words in top 5"], table
        ),
        "",
        f"![chunking]({link})",
        "",
    ]
    return lines, rows


def run(cfg: SearchConfig) -> str:
    set_seed(cfg.seed)
    out = Outputs(cfg.output_dir, cfg.experiment)
    sections: list[str] = []
    rows: list[dict[str, Any]] = []
    with track_run(cfg.experiment, cfg.model_dump(mode="json"), cfg.tags) as tracker:
        for name in cfg.datasets:
            qa = load_qa(name, n_questions=cfg.n_questions, seed=cfg.seed)
            lines, ds_rows = run_retrieval(qa, cfg)
            sections += lines
            rows += ds_rows
            if name == cfg.chunk_dataset:
                lines, ch_rows = run_chunking(qa, cfg, out)
                sections += lines
                rows += ch_rows
        frame = write_results(rows, out.results)
        best = frame[frame.metric == "ndcg@10"].groupby(["dataset", "model"]).value.mean()
        tracker.log_metrics({f"{d}.{m}.ndcg10": v for (d, m), v in best.items()})
        header = [
            "# B3 — Semantic search: BM25 vs. dense vs. hybrid vs. re-ranking",
            "",
            "Dense retrievers use normalized embeddings in a FAISS inner-product index; hybrid = "
            "reciprocal-rank fusion (k = 60) of BM25 and the dense run; re-ranking = cross-encoder "
            "(`ms-marco-MiniLM-L-6-v2`) over the first stage's top 20. Significance: per-query "
            f"nDCG@10, Wilcoxon signed-rank vs. the best retriever, Holm-adjusted. Per-query results in [`results/{cfg.experiment}.csv`](results/{cfg.experiment}.csv).",
            "",
        ]
        text = "\n".join(header + sections)
        out.write_report(text)
        tracker.log_artifact(out.report)
    return text


def main() -> None:
    cli(run, SearchConfig, "B3 semantic search")


if __name__ == "__main__":
    main()
