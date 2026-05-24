# ScholarRAG

**🚀 [Live Demo](https://huggingface.co/spaces/akashiitb/scholar-rag)** &nbsp;·&nbsp; Hybrid Retrieval · Cross-Encoder Reranking · Grounded Generation

A retrieval-augmented assistant that answers natural-language questions over a
**continuously-updated corpus of arXiv papers** (cs.CV, cs.LG, cs.CL) and cites
the exact papers it used. Built to stay current with ML research — the kind of
tool a researcher actually wants.

**Stack:** Python · BGE-M3 (dense + sparse) · Qdrant · BGE cross-encoder reranker ·
Llama 3.3 70B (Groq) · FastAPI · Gradio · SLURM (GPU indexing) · HuggingFace Spaces

| | |
|---|---|
| **Live demo** | https://huggingface.co/spaces/akashiitb/scholar-rag |
| **Retrieval** | Hybrid (BGE-M3 dense + sparse), RRF fusion, top-50 |
| **Reranking** | `bge-reranker-v2-m3` cross-encoder, top-8 |
| **Generation** | Llama 3.3 70B via Groq, grounded + cited, refusal guardrail |
| **Freshness** | Daily incremental arXiv ingestion (SLURM job on GPU cluster) |

> **Why RAG and not a fine-tuned model?** New papers are published every day. An
> answer about "the latest work on X" must be *current* and must *cite a real
> source*. A fine-tuned model goes stale the moment training ends and invents
> plausible-but-fake citations. RAG keeps knowledge fresh (just ingest new
> papers) and grounded (every claim points to a retrievable source). That
> freshness requirement is the entire justification for this architecture.

---

## What it does

- Pulls new arXiv papers **daily** and indexes them (incremental upsert).
- Answers questions like *"What are recent approaches to test-time adaptation
  for segmentation?"* with a grounded summary and links to the source papers.
- Supports metadata filters: *"papers after Jan 2026 in cs.CV on diffusion models."*
- Returns citations so every claim is verifiable.

## Architecture

Two lanes that run at different times:

**Indexing lane** (daily, on the GPU compute node):
`arXiv API → chunk → BGE-M3 embed (dense + sparse) → Qdrant upsert`

**Query lane** (on demand, deployable):
`question → FastAPI → hybrid retrieve (top-50) → bge-reranker (top-8) → Groq LLM → cited answer → Gradio UI`

| Component        | Choice                          | Why                                                            |
|------------------|---------------------------------|----------------------------------------------------------------|
| Data source      | arXiv API                       | Free, public, no key, updates daily                            |
| Embeddings       | `BAAI/bge-m3`                   | One model gives dense **and** sparse vectors → native hybrid   |
| Vector DB        | Qdrant                          | Hybrid search, payload filtering (date/category), fast upserts |
| Reranker         | `BAAI/bge-reranker-v2-m3`       | Lightweight cross-encoder, big precision lift, runs on GPU     |
| LLM (generation) | Groq `llama-3.3-70b` (free)     | Fast, free tier; swappable for local vLLM on the cluster       |
| Serving          | FastAPI + Gradio                | Streaming API + simple demo UI                                 |
| Evaluation       | RAGAS + retrieval metrics       | Measures retrieval and generation **separately**               |
| Deployment       | HF Spaces + Qdrant Cloud (free) | Public live demo at zero cost                                  |

## Project structure

```
scholar-rag/
├── README.md
├── requirements.txt
├── .env.example                  # copy to .env, fill in keys
├── config.yaml                   # categories, chunk size, top-k, model names
├── data/
│   ├── raw/                      # fetched arXiv metadata (jsonl)
│   └── golden/                   # golden eval Q/A set
├── src/
│   ├── config.py                 # loads config.yaml + .env
│   ├── ingest/
│   │   ├── arxiv_client.py       # fetch papers by category + date window
│   │   └── chunker.py            # section-aware + contextual chunking
│   ├── index/
│   │   ├── embedder.py           # BGE-M3 dense + sparse encoding
│   │   └── vector_store.py       # Qdrant create / upsert / hybrid search
│   ├── retrieve/
│   │   ├── retriever.py          # hybrid retrieval (dense + sparse fusion)
│   │   └── reranker.py           # bge-reranker-v2-m3 cross-encoder
│   ├── generate/
│   │   ├── prompt.py             # grounded prompt + citation format
│   │   └── llm.py                # Groq client + anti-hallucination guardrail
│   ├── pipeline.py               # orchestrates question → cited answer
│   └── api/
│       ├── server.py             # FastAPI app (streaming)
│       └── ui.py                 # Gradio frontend
├── scripts/
│   ├── setup_env.sh              # conda env creation on the cluster
│   ├── index_papers.slurm        # SLURM: full index build (GPU)
│   ├── daily_update.slurm        # SLURM: daily incremental upsert
│   └── build_golden_set.py       # generate eval Q/A pairs
├── eval/
│   ├── eval_retrieval.py         # recall@k, MRR, nDCG
│   ├── eval_ragas.py             # faithfulness, answer relevancy, context P/R
│   └── results/                  # eval reports (committed, for the README)
├── notebooks/
│   └── 00_concepts.ipynb         # Phase 0 learning notebook
├── tests/
│   └── test_retriever.py
└── Dockerfile                    # for HF Spaces deployment
```

---

## Setup

### 1. Free accounts / keys

| Service     | Cost | What you need                                                      |
|-------------|------|--------------------------------------------------------------------|
| arXiv API   | free | nothing (public)                                                   |
| HuggingFace | free | account + access token (to download BGE-M3 + reranker weights)     |
| Groq        | free | account + API key (generous free tier)                             |
| Qdrant      | free | local Docker for dev; Qdrant Cloud free tier (1 GB) for the demo   |

Copy `.env.example` to `.env` and fill in:

```bash
HF_TOKEN=hf_xxx
GROQ_API_KEY=gsk_xxx
QDRANT_URL=http://localhost:6333        # or your Qdrant Cloud URL
QDRANT_API_KEY=                          # blank for local
```

### 2. Environment on the university cluster

Run on the **login node** (no GPU needed for setup):

```bash
git clone <your-repo-url> scholar-rag && cd scholar-rag
bash scripts/setup_env.sh        # creates a conda env `scholarrag` and installs deps
```

Model weights (BGE-M3 ~2.2 GB, reranker ~600 MB) download on first use into your
HF cache. Set `HF_HOME` to a project/scratch dir so you don't fill your home quota:

```bash
export HF_HOME=/scratch/$USER/hf_cache
```

---

## Running it

### Build the index (GPU — submit as a SLURM batch job)

The login node has no GPU, so all embedding/reranking runs on a compute node via
SLURM. **Never run model inference on the login node.**

```bash
sbatch scripts/index_papers.slurm        # builds the initial index
squeue -u $USER                          # check job status
tail -f logs/index_*.out                 # watch progress
```

### Keep it fresh (daily incremental update)

```bash
sbatch scripts/daily_update.slurm        # fetch yesterday's papers, upsert
```

To automate, add a cron entry on the login node that submits the job nightly:

```bash
# crontab -e  (on the login node)
0 6 * * *  cd /path/to/scholar-rag && sbatch scripts/daily_update.slurm
```

### Serve the app

For development, request an interactive GPU session, then:

```bash
# inside an interactive compute-node session (e.g. srun --gres=gpu:1 --pty bash)
conda activate scholarrag
uvicorn src.api.server:app --host 0.0.0.0 --port 8000     # API
python -m src.api.ui                                       # Gradio UI
```

For the **public demo**, deploy the query lane to HuggingFace Spaces pointing at
Qdrant Cloud (the index built on the cluster is pushed to Qdrant Cloud). See
`Dockerfile`.

---

## Evaluation

A RAG system has two failure modes that must be measured **separately**:
retrieval failure (wrong chunks fetched) and generation failure (right chunks,
wrong answer). We measure both against a golden set of ~150 Q/A pairs built from
the corpus (`scripts/build_golden_set.py`, then manually verified).

### Retrieval metrics — `eval/eval_retrieval.py`

| Metric     | Question it answers                                         |
|------------|-------------------------------------------------------------|
| Recall@k   | Did the relevant chunk make it into the top-k?              |
| MRR        | How high up was the first relevant chunk?                   |
| nDCG@k     | Are the most relevant chunks ranked near the top?           |

### Generation metrics — `eval/eval_ragas.py` (RAGAS)

| Metric            | Question it answers                                       |
|-------------------|-----------------------------------------------------------|
| Faithfulness      | Is every claim supported by the retrieved context?       |
| Answer relevancy  | Does the answer actually address the question?           |
| Context precision | Are the retrieved chunks relevant (low noise)?           |
| Context recall    | Did we retrieve everything needed to answer?             |

### Ablations to report (this is what impresses interviewers)

Run the eval with each config and put the table in this README:

1. Dense-only vs hybrid retrieval → show the Recall@5 lift.
2. No reranker vs reranker → show the nDCG / faithfulness lift.
3. Chunk size sweep (256 / 512 / 1024 tokens) → show the sweet spot.

> The point isn't a high score — it's that you can say *"naive setup gave X,
> these changes gave Y, and here's exactly where it still fails."* That honesty
> plus numbers is what gets offers.

---

## Roadmap

- [x] Phase 0 — concepts (notebook)
- [ ] Phase 1 — ingest + chunk arXiv papers
- [ ] Phase 2 — index + hybrid retrieval + reranking + golden set
- [ ] Phase 3 — grounded generation + guardrails + RAGAS
- [ ] Phase 4 — FastAPI + Gradio + Docker
- [ ] Phase 5 — deploy to HF Spaces + write up eval numbers

## License

MIT
