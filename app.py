import os
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
from src.api.ui import demo

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        show_api=False,
    )
