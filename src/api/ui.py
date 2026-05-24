"""Gradio UI for ScholarRAG — the interactive demo interviewers will use.

Features:
  - Streaming answer (tokens appear live, feels fast and modern)
  - Example questions to click (so a visitor isn't staring at a blank box)
  - Sources panel rendering the cited papers with arXiv links + relevance scores
  - Clean, professional research-tool aesthetic

Runs entirely on CPU at the user's end — the heavy embedding model loads once at
startup, retrieval hits Qdrant Cloud, and generation happens on Groq's servers.
This is what gets deployed to HuggingFace Spaces.

Run locally:   python -m src.api.ui
On HF Spaces:  the Space runs this file as `app.py` (see deployment guide).
"""
from __future__ import annotations

import gradio as gr

from src.retrieve.retriever import retrieve
from src.generate.llm import generate_stream


EXAMPLES = [
    "What recent methods explore test-time adaptation or search?",
    "Summarize recent work on video understanding with large language models.",
    "What are new approaches to tokenization in NLP?",
    "Recent advances in vision-language navigation for robots?",
]


def _format_sources_md(results: list[dict]) -> str:
    """Render retrieved papers as a markdown citation list."""
    if not results:
        return "_No sources retrieved._"
    lines = ["### Sources\n"]
    for i, r in enumerate(results, 1):
        p = r["payload"]
        title = p.get("title", "Unknown")
        url = p.get("abs_url", "")
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        date = p.get("published", "")[:10]
        score = round(r.get("rerank_score", r.get("score", 0)), 4)
        lines.append(
            f"**[{i}] [{title}]({url})**  \n"
            f"<sub>{authors} · {date} · relevance {score}</sub>\n"
        )
    return "\n".join(lines)


def respond(message: str, history):
    """Streaming generator for the chat. Yields partial answers as they arrive,
    then appends the sources block at the end."""
    if not message or not message.strip():
        yield "Please enter a question about recent ML papers."
        return

    # Retrieve once (this also loads models on first call)
    results = retrieve(message)

    # Stream the grounded answer token by token
    partial = ""
    for token in generate_stream(message, results):
        partial += token
        yield partial

    # Append sources after the answer completes
    sources_md = _format_sources_md(results)
    yield partial + "\n\n---\n\n" + sources_md


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="ScholarRAG") as demo:
    gr.Markdown(
        """
        # 📚 ScholarRAG
        Ask questions about **recent machine-learning papers** (cs.CV / cs.LG / cs.CL).
        Answers are generated *only* from retrieved arXiv papers, with citations —
        if the indexed papers don't contain the answer, the assistant says so instead
        of guessing.

        <sub>Hybrid retrieval (BGE-M3 dense + sparse) → cross-encoder reranking →
        grounded generation (Llama 3.3 70B via Groq). Index updates daily.</sub>
        """
    )

    gr.ChatInterface(
        fn=respond,
        examples=EXAMPLES,
        chatbot=gr.Chatbot(height=480, show_copy_button=True, type="messages"),
        textbox=gr.Textbox(
            placeholder="Ask about recent ML research…",
            container=False,
            scale=7,
        ),
        type="messages",
    )

    gr.Markdown(
        "<sub>Built by Akash Pal · "
        "[GitHub](https://github.com/Akashpal6060/scholar-rag)</sub>"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
