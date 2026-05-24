---
title: ScholarRAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# ScholarRAG

A retrieval-augmented assistant that answers questions about recent arXiv ML
papers (cs.CV / cs.LG / cs.CL) with citations. Answers are grounded only in
retrieved papers — if the index lacks the answer, it says so instead of guessing.

**Pipeline:** BGE-M3 hybrid retrieval (dense + sparse) → cross-encoder reranking
→ grounded generation (Llama 3.3 70B via Groq). Vectors served from Qdrant Cloud;
index refreshed daily.

Built by Akash Pal · [GitHub](https://github.com/Akashpal6060/scholar-rag)
