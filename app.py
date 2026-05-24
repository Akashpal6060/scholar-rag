"""HuggingFace Spaces entrypoint.

HF Spaces looks for `app.py` at the repo root and runs it. This re-exports the
Gradio demo from src/api/ui.py. On Spaces, share=False (HF gives the public URL).
"""
import os

# HF Spaces sets a writable cache dir; point HF_HOME there before model loads.
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

from src.api.ui import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
